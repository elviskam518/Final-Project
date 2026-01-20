import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


def make_group(df: pd.DataFrame,
               gender_col="Gender",
               race_col="Race",
               group_col="Group") -> pd.DataFrame:
    df = df.copy()
    df[group_col] = df[gender_col].astype(str) + "_" + df[race_col].astype(str)
    return df


COLUMN_MAPPING = {
    "Gender": "Gender",
    "Race": "Race",
    "ExperienceYears": "YearsExperience",
    "EducationLevel": "EducationLevel",
    "SkillScore": "AlgorithmSkill",
    "InterviewScore": "OverallInterviewScore",
    "Hired": "Hired"
}

TECH_FEATURE_COLS = {
    "experience": "YearsExperience",
    "education": "EducationLevel",
    "skill": "AlgorithmSkill",
    "interview": "OverallInterviewScore"
}


def load_tech_data(csv_path="tech_diversity_hiring_data.csv"):
    df = pd.read_csv(csv_path)

    print("=" * 60)
    print("Loaded Tech Industry Diversity Hiring Dataset")
    print("=" * 60)
    print(f"Number of samples: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    return df


def add_proxy_qualified_tech(
    df: pd.DataFrame,
    experience_col="YearsExperience",
    education_col="EducationLevel",
    skill_col="AlgorithmSkill",
    interview_col="OverallInterviewScore",
    score_col="QualificationScore",
    qualified_col="Qualified",
    top_quantile=0.40,
    weights=(0.15, 0.15, 0.35, 0.35)
) -> pd.DataFrame:

    df = df.copy()

    for c in [experience_col, education_col, skill_col, interview_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    w_exp, w_edu, w_skill, w_int = weights

    score = (
        w_exp * zscore(df[experience_col]) +
        w_edu * zscore(df[education_col]) +
        w_skill * zscore(df[skill_col]) +
        w_int * zscore(df[interview_col])
    )

    df[score_col] = score
    cutoff = df[score_col].quantile(1 - top_quantile)
    df[qualified_col] = (df[score_col] >= cutoff).astype(int)

    print("\nProxy Qualified Setup:")
    print(f"  Weights: experience={w_exp}, education={w_edu}, skill={w_skill}, interview={w_int}")
    print(f"  Top {top_quantile*100:.0f}% labeled as qualified")
    print(f"  Qualified count: {df[qualified_col].sum():,} ({df[qualified_col].mean()*100:.1f}%)")

    return df


def compute_proxy_eo_by_group(
    df: pd.DataFrame,
    decision_col="Hired",
    gender_col="Gender",
    race_col="Race",
    baseline_gender="Male",
    baseline_race="White",
    qualified_col="Qualified",
    group_col="Group",
) -> pd.DataFrame:

    df = df.copy()

    for c in [decision_col, gender_col, race_col, qualified_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df = make_group(df, gender_col, race_col, group_col)
    baseline_group = f"{baseline_gender}_{baseline_race}"

    grp = df.groupby(group_col)

    out = grp.agg(
        group_size=(decision_col, "size"),
        hire_rate=(decision_col, "mean"),
        qualified_rate=(qualified_col, "mean"),
    ).reset_index()

    tpr_rows = []
    for g, gdf in df.groupby(group_col):
        qdf = gdf[gdf[qualified_col] == 1]
        tpr = np.nan if len(qdf) == 0 else qdf[decision_col].mean()
        tpr_rows.append((g, tpr, len(qdf)))

    tpr_df = pd.DataFrame(tpr_rows, columns=[group_col, "TPR_proxy", "qualified_count"])
    out = out.merge(tpr_df, on=group_col, how="left")

    base_row = out[out[group_col] == baseline_group]
    if len(base_row) == 0:
        print(f"Warning: Baseline group '{baseline_group}' not found")
        return out

    base_hire_rate = float(base_row["hire_rate"].iloc[0])
    base_tpr = float(base_row["TPR_proxy"].iloc[0])

    out["DI_vs_baseline"] = out["hire_rate"] / base_hire_rate if base_hire_rate > 0 else np.nan
    out["EO_gap_vs_baseline"] = out["TPR_proxy"] - base_tpr

    out["is_baseline"] = (out[group_col] == baseline_group).astype(int)
    out = out.sort_values(["is_baseline", group_col], ascending=[False, True])
    out = out.drop(columns=["is_baseline"])

    return out


def run_intersectional_logit_or(
    df: pd.DataFrame,
    decision_col="Hired",
    group_col="Group",
    baseline_group="Male_White",
    experience_col="YearsExperience",
    education_col="EducationLevel",
    skill_col="AlgorithmSkill",
    interview_col="OverallInterviewScore",
    output_csv="tech_logit_oddsratio_by_group.csv"
) -> pd.DataFrame:

    df = df.copy()

    needed = [decision_col, group_col, experience_col, education_col, skill_col, interview_col]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    feature_cols = [experience_col, education_col, skill_col, interview_col]

    group_dummies = pd.get_dummies(df[group_col], prefix="G", drop_first=False)
    baseline_col = f"G_{baseline_group}"
    if baseline_col in group_dummies.columns:
        group_dummies = group_dummies.drop(columns=[baseline_col])

    scaler = StandardScaler()
    X_features = scaler.fit_transform(df[feature_cols].values)
    X = np.hstack([X_features, group_dummies.values])
    y = df[decision_col].values

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X, y)

    feature_names = feature_cols + list(group_dummies.columns)
    coefs = model.coef_[0]

    rows = []
    for i, name in enumerate(feature_names):
        coef = coefs[i]
        or_val = np.exp(coef)

        if name.startswith("G_"):
            group_name = name[2:]
            if or_val > 1:
                interp = f"Hiring odds {((or_val - 1) * 100):.1f}% higher than {baseline_group}"
            elif or_val < 1:
                interp = f"Hiring odds {((1 - or_val) * 100):.1f}% lower than {baseline_group}"
            else:
                interp = f"Same as {baseline_group}"
        else:
            if or_val > 1:
                interp = f"Per +1 SD, hiring odds increase {((or_val - 1) * 100):.1f}%"
            elif or_val < 1:
                interp = f"Per +1 SD, hiring odds decrease {((1 - or_val) * 100):.1f}%"
            else:
                interp = "No effect"

        rows.append({
            "term": name,
            "coef": coef,
            "odds_ratio": or_val,
            "interpretation": interp
        })

    rows.append({
        "term": "Intercept",
        "coef": model.intercept_[0],
        "odds_ratio": np.exp(model.intercept_[0]),
        "interpretation": "Baseline hiring odds"
    })

    out = pd.DataFrame(rows).sort_values("term").reset_index(drop=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("Intersectional Logistic Regression Results")
    print("=" * 70)
    print(f"Baseline group: {baseline_group}")
    print(f"Model accuracy: {model.score(X, y):.4f}")
    print(f"Saved to: {output_csv}")

    group_terms = out[out["term"].str.startswith("G_")]
    print("\nGroup-level Odds Ratios:")
    print(group_terms[["term", "coef", "odds_ratio", "interpretation"]].round(4).to_string(index=False))

    return out


def generate_fairness_report(df: pd.DataFrame, eo_results: pd.DataFrame, logit_results: pd.DataFrame):

    print("\n" + "=" * 70)
    print("Tech Hiring Fairness Analysis Report")
    print("=" * 70)

    print("\n1. Disparate Impact Analysis")
    print("-" * 50)
    print("DI < 0.8 indicates potential discrimination\n")

    di_issues = eo_results[eo_results["DI_vs_baseline"] < 0.8]
    if len(di_issues) > 0:
        for _, row in di_issues.iterrows():
            print(f"{row['Group']}: DI = {row['DI_vs_baseline']:.4f}")
    else:
        print("All groups satisfy DI ≥ 0.8")

    print("\n2. Equal Opportunity Analysis")
    print("-" * 50)

    eo_issues = eo_results[eo_results["EO_gap_vs_baseline"].abs() > 0.10]
    if len(eo_issues) > 0:
        for _, row in eo_issues.iterrows():
            direction = "lower" if row["EO_gap_vs_baseline"] < 0 else "higher"
            print(f"{row['Group']}: TPR {direction} than baseline by {abs(row['EO_gap_vs_baseline'])*100:.1f}%")

   



if __name__ == "__main__":
    df = load_tech_data("tech_diversity_hiring_data.csv")

    df = add_proxy_qualified_tech(
        df,
        experience_col="YearsExperience",
        education_col="EducationLevel",
        skill_col="AlgorithmSkill",
        interview_col="OverallInterviewScore"
    )

    eo_result = compute_proxy_eo_by_group(
        df,
        decision_col="Hired",
        gender_col="Gender",
        race_col="Race",
        baseline_gender="Male",
        baseline_race="White"
    )

    print("\n" + "=" * 70)
    print("Proxy EO and DI Results")
    print("=" * 70)
    print(eo_result.round(4).to_string(index=False))
    eo_result.to_csv("tech_proxy_eo_group_metrics.csv", index=False, encoding="utf-8-sig")

    df = make_group(df, "Gender", "Race")

    logit_or = run_intersectional_logit_or(
        df,
        decision_col="Hired",
        group_col="Group"
    )

    generate_fairness_report(df, eo_result, logit_or)

    print("\nAnalysis completed.")
