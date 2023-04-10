import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import torch

df = pd.read_csv("Mental Health Data.csv")

a = list(df.columns)
b = [
    "self_employed",
    "no_of_employees",
    "tech_company",
    "role_IT",
    "mental_healthcare_coverage",
    "knowledge_about_mental_healthcare_options_workplace",
    "employer_discussed_mental_health ",
    "employer_offer_resources_to_learn_about_mental_health",
    "medical_leave_from_work ",
    "comfortable_discussing_with_coworkers",
    "employer_take_mental_health_seriously",
    "knowledge_of_local_online_resources ",
    "productivity_affected_by_mental_health ",
    "percentage_work_time_affected_mental_health",
    "openess_of_family_friends",
    "family_history_mental_illness",
    "mental_health_disorder_past",
    "currently_mental_health_disorder",
    "diagnosed_mental_health_condition",
    "type_of_disorder",
    "treatment_from_professional",
    "while_effective_treatment_mental_health_issue_interferes_work",
    "while_not_effective_treatment_interferes_work ",
    "age",
    "gender",
    "country",
    "US state",
    "country work ",
    "US state work",
    "role_in_company",
    "work_remotely",
]

for i, j in zip(a, b):
    df.rename(columns={i: j}, inplace=True)


cols = [
    "role_IT",
    "knowledge_of_local_online_resources ",
    "productivity_affected_by_mental_health ",
    "percentage_work_time_affected_mental_health",
    "type_of_disorder",
    "US state",
    "US state work",
]

df = df.drop(cols, axis=1)

df["no_of_employees"].replace(
    to_replace=["1 to 5", "6 to 25", "More than 1000", "26-99"],
    value=["1-5", "6-25", ">1000", "26-100"],
    inplace=True,
)

df["mental_healthcare_coverage"].replace(
    to_replace=["Not eligible for coverage / N/A"], value="No", inplace=True
)

df["openess_of_family_friends"].replace(
    to_replace=["Not applicable to me (I do not have a mental illness)"],
    value="I don't know",
    inplace=True,
)

med_age = df[(df["age"] >= 18) | (df["age"] <= 75)]["age"].median()
df["age"].replace(
    to_replace=df[(df["age"] < 18) | (df["age"] > 75)]["age"].tolist(),
    value=med_age,
    inplace=True,
)
# Define the scaler object
scaler = MinMaxScaler()

# Fit and transform the 'age' column
df["age"] = scaler.fit_transform(df[["age"]])


df["gender"].replace(
    to_replace=[
        "Male",
        "male",
        "Male ",
        "M",
        "m",
        "man",
        "Cis male",
        "Male.",
        "male 9:1 female, roughly",
        "Male (cis)",
        "Man",
        "Sex is male",
        "cis male",
        "Malr",
        "Dude",
        "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
        "mail",
        "M|",
        "Male/genderqueer",
        "male ",
        "Cis Male",
        "Male (trans, FtM)",
        "cisdude",
        "cis man",
        "MALE",
    ],
    value="male",
    inplace=True,
)
df["gender"].replace(
    to_replace=[
        "Female",
        "female",
        "I identify as female.",
        "female ",
        "Female assigned at birth ",
        "F",
        "Woman",
        "fm",
        "f",
        "Cis female ",
        "Transitioned, M2F",
        "Genderfluid (born female)",
        "Female or Multi-Gender Femme",
        "Female ",
        "woman",
        "female/woman",
        "Cisgender Female",
        "fem",
        "Female (props for making this a freeform field, though)",
        " Female",
        "Cis-woman",
        "female-bodied; no feelings about gender",
        "AFAB",
    ],
    value="female",
    inplace=True,
)
df["gender"].replace(
    to_replace=[
        "Bigender",
        "non-binary",
        "Other/Transfeminine",
        "Androgynous",
        "Other",
        "nb masculine",
        "none of your business",
        "genderqueer",
        "Human",
        "Genderfluid",
        "Enby",
        "genderqueer woman",
        "mtf",
        "Queer",
        "Agender",
        "Fluid",
        "Nonbinary",
        "human",
        "Unicorn",
        "Genderqueer",
        "Genderflux demi-girl",
        "Transgender woman",
    ],
    value="other",
    inplace=True,
)

tech_list = []
tech_list.append(
    df[df["role_in_company"].str.contains("Back-end")]["role_in_company"].tolist()
)
tech_list.append(
    df[df["role_in_company"].str.contains("Front-end")]["role_in_company"].tolist()
)
tech_list.append(
    df[df["role_in_company"].str.contains("Dev")]["role_in_company"].tolist()
)
tech_list.append(
    df[df["role_in_company"].str.contains("DevOps")]["role_in_company"].tolist()
)
flat_list = [item for sublist in tech_list for item in sublist]
flat_list = list(dict.fromkeys(flat_list))

df["tech_role"] = df["role_in_company"]
df["tech_role"].replace(to_replace=flat_list, value=1, inplace=True)
remain_list = df["tech_role"].unique()[1:]
df["tech_role"].replace(to_replace=remain_list, value=0, inplace=True)


df = df.drop(["role_in_company"], axis=1)


imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
imp.fit(df)
df = pd.DataFrame(data=imp.transform(df), columns=df.columns)

df["age"] = df["age"].astype(float)


X = df.drop(
    [
        "diagnosed_mental_health_condition",
        "treatment_from_professional",
        "while_effective_treatment_mental_health_issue_interferes_work",
        "while_not_effective_treatment_interferes_work ",
    ],
    axis=1,
)

feature_cols = [
    "self_employed",
    "no_of_employees",
    "tech_company",
    "mental_healthcare_coverage",
    "knowledge_about_mental_healthcare_options_workplace",
    "employer_discussed_mental_health ",
    "employer_offer_resources_to_learn_about_mental_health",
    "medical_leave_from_work ",
    "comfortable_discussing_with_coworkers",
    "employer_take_mental_health_seriously",
    "openess_of_family_friends",
    "family_history_mental_illness",
    "mental_health_disorder_past",
    "currently_mental_health_disorder",
    "gender",
    "country",
    "country work ",
    "work_remotely",
    "tech_role",
]

X = pd.get_dummies(X, columns=feature_cols)
x = X.to_numpy().astype(np.float64)
y = df["diagnosed_mental_health_condition"].replace({"No": 0, "Yes": 1}).to_numpy()


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1
)

y_train = y_train.reshape((859, 1))
y_val = y_val.reshape((287, 1))
y_test = y_test.reshape((287, 1))


def convert_to_tensor(data):
    return torch.from_numpy(data).type(torch.float64)


train_inputs_pt = convert_to_tensor(X_train)
train_outputs_pt = convert_to_tensor(y_train)

val_inputs_pt = convert_to_tensor(X_val)
val_outputs_pt = convert_to_tensor(y_val)

test_inputs_pt = convert_to_tensor(X_test)
test_outputs_pt = convert_to_tensor(y_test)