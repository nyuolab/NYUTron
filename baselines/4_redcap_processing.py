import pandas as pd
from random import shuffle


def generate_questions(pos_path, neg_path):
    df_pos = pd.read_csv(pos_path)
    df_neg = pd.read_csv(neg_path)
    cases = []  # list of cases, 100 cases in general
    # go through readmitted cases to generate questions
    for _, row in df_pos.iterrows():
        dc = row["discharge_summary"]
        hp = row["hp_note"]
        q1 = f"Will this person be readmitted within 30 days? \n{dc}"
        q2 = f"Is this readmission related to prior discharge?\n{hp}"
        q3 = "Is this readmission preventable?"
        q4 = "Any comments? (how to prevent this readmission? Why is prevention impossible?)"
    qs = [q1, q2, q3, q4]  # list of questions associated with the case
    cases.append(qs)
    # go through non-readmitted cases to generate questions
    for _, row in df_neg.iterrows():
        dc = row["discharge_summary"]
        q1 = f"Will this person be readmitted within 30 days? \n{dc}"
        qs = [q1]  # only 1 question associated with non-readmitted case
        cases.append(qs)
    # shuffle the sequence of cases
    shuffle(cases)
    # check order of shuffled cases
    # len 1: non-readmitted case
    # len 4: readmitted case
    lens = []
    for idx in range(30):
        lens.append(len(cases[idx]))
    # organize the shuffled cases into redcap questions
    redcap_qs = []
    for i in range(len(cases)):
        case = cases[i]
        for q in case:
            prefix = f"cases {i+1}: "
            redcap_qs.append(prefix + q)
    return pd.DataFrame(redcap_qs)


def update_redcap_survey(redcap_csv_path):
    qs = pd.read_csv(redcap_csv_path)
    # go through readmitted cases to generate questions
    for index, row in qs.iterrows():
        qtext = row["Field Label"]
        if "cases" in qtext:
            # add definition of CMS definition of readmission
            qs["Section Header"][
                index
            ] = "CMS defines a hospital readmission as an admission to an acute care hospital within 30 days of discharge from the same or another acute care hospital."
        if "Is this readmission related" in qtext:
            # add additional choices evaluation
            choices = row["Choices, Calculations, OR Slider Labels"]
            choices += (
                " (explain) | 3, Does Not Meet Medicare Criteria for 30d re-admission"
            )
            qs["Choices, Calculations, OR Slider Labels"][index] = choices
    return qs
