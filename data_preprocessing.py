import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import torch

df = pd.read_csv("Mental Health Data.csv")

# This code is renaming the columns of the pandas DataFrame `df` to match the desired column names
# listed in the `b` list. It first creates a list of the current column names using `list(df.columns)`
# and then uses `zip` to iterate through both lists (`a` and `b`) simultaneously and rename the
# columns using the `rename` method of the DataFrame.
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


# This code is dropping certain columns from the pandas DataFrame `df`. The columns to be dropped are
# specified in the `cols` list. The `drop` method is then used on the DataFrame
# to remove these columns along the specified axis (axis=1, which refers to columns). The resulting
# DataFrame `df` will no longer contain these columns.
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

# This code is replacing the values in the "no_of_employees" column of the pandas DataFrame `df`. The
# values to be replaced are specified in the `to_replace` list, and the corresponding replacement
# values are specified in the `value` list. The `replace` method is used on the column, with the
# `inplace` parameter set to `True` to modify the DataFrame in place.
df["no_of_employees"].replace(
    to_replace=["1 to 5", "6 to 25", "More than 1000", "26-99"],
    value=["1-5", "6-25", ">1000", "26-100"],
    inplace=True,
)

# This line of code is replacing the value "Not eligible for coverage / N/A" in the
# "mental_healthcare_coverage" column of the pandas DataFrame `df` with the value "No". The
# `inplace=True` parameter ensures that the DataFrame is modified in place. This is likely done to
# standardize the values in the column and make it easier to analyze or model the data.
df["mental_healthcare_coverage"].replace(
    to_replace=["Not eligible for coverage / N/A"], value="No", inplace=True
)

# This code is replacing the value "Not applicable to me (I do not have a mental illness)" in the
# "openess_of_family_friends" column of the pandas DataFrame `df` with the value "I don't know". The
# `inplace=True` parameter ensures that the DataFrame is modified in place.
df["openess_of_family_friends"].replace(
    to_replace=["Not applicable to me (I do not have a mental illness)"],
    value="I don't know",
    inplace=True,
)

# This code is performing data preprocessing on the "age" column of the pandas DataFrame `df`.
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


# This code is standardizing the values in the "gender" column of the pandas DataFrame `df`. It
# replaces various forms of male and female gender labels with "male" and "female", respectively, and
# replaces various non-binary gender labels with "other". This is likely done to make the data easier
# to analyze or model, as it reduces the number of unique values in the column. The `inplace=True`
# parameter ensures that the DataFrame is modified in place.
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

# This code is creating a new column in the pandas DataFrame `df` called "tech_role". It first creates
# a list of all the unique values in the "role_in_company" column that contain the strings "Back-end",
# "Front-end", "Dev", or "DevOps". It then replaces these values in the new "tech_role" column with
# the value 1, and all other values with the value 0. This is likely done to create a binary feature
# that indicates whether an individual's role in the company is related to technology or not, which
# could be useful in predicting their likelihood of being diagnosed with a mental health condition.
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

# This code is creating a new column in the pandas DataFrame `df` called "tech_role". It first copies
# the values from the "role_in_company" column to the new "tech_role" column. It then replaces the
# values in the "tech_role" column that contain the strings "Back-end", "Front-end", "Dev", or
# "DevOps" with the value 1, and all other values with the value 0. This is likely done to create a
# binary feature that indicates whether an individual's role in the company is related to technology
# or not, which could be useful in predicting their likelihood of being diagnosed with a mental health
# condition.
df["tech_role"] = df["role_in_company"]
df["tech_role"].replace(to_replace=flat_list, value=1, inplace=True)
remain_list = df["tech_role"].unique()[1:]
df["tech_role"].replace(to_replace=remain_list, value=0, inplace=True)


df = df.drop(["role_in_company"], axis=1)


# This code is performing imputation on the pandas DataFrame `df` using the `SimpleImputer` class from
# scikit-learn. The `missing_values` parameter is set to `np.nan`, indicating that missing values in
# the DataFrame are represented by NaN values. The `strategy` parameter is set to `"most_frequent"`,
# indicating that missing values should be imputed with the most frequent value in each column.
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

# `X = pd.get_dummies(X, columns=feature_cols)` is creating dummy variables for categorical features
# in the DataFrame `X` using one-hot encoding. The `columns` parameter specifies which columns to
# encode.
X = pd.get_dummies(X, columns=feature_cols)
x = X.to_numpy().astype(np.float64)
y = df["diagnosed_mental_health_condition"].replace({"No": 0, "Yes": 1}).to_numpy()


# This code is splitting the dataset into three sets: a training set, a validation set, and a test
# set.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1
)

y_train = y_train.reshape((859, 1))
y_val = y_val.reshape((287, 1))
y_test = y_test.reshape((287, 1))


def convert_to_tensor(data):
    """
    The function converts numpy arrays to PyTorch tensors of type float64.
    
    :param data: "data" refers to the input data that needs to be converted to PyTorch tensors. In this
    case, the input data is split into training, validation, and test sets, and each set is converted to
    PyTorch tensors using the "convert_to_tensor" function. The resulting tensors are then
    :return: The function `convert_to_tensor` takes in a numpy array `data` and returns a PyTorch tensor
    of type `torch.float64` created from the input data. The code then uses this function to convert the
    training, validation, and test data from numpy arrays to PyTorch tensors. The returned values are
    the PyTorch tensors `train_inputs_pt`, `train_outputs_pt`, `
    """
    return torch.from_numpy(data).type(torch.float64)


train_inputs_pt = convert_to_tensor(X_train)
train_outputs_pt = convert_to_tensor(y_train)

val_inputs_pt = convert_to_tensor(X_val)
val_outputs_pt = convert_to_tensor(y_val)

test_inputs_pt = convert_to_tensor(X_test)
test_outputs_pt = convert_to_tensor(y_test)