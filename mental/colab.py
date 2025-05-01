import pandas as pd
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


file_path = r"C:\Users\lenovo\OneDrive\Desktop\mental\dataset.csv"
df = pd.read_csv(file_path)


selected_columns = df.columns[:-1]
missing_columns = set(selected_columns) - set(df.columns)
if missing_columns:
    print("Error: The following columns are not present in the dataset:", missing_columns)
else:
    X = df[selected_columns]
    y = df["Disorder"]
    categorical_columns = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_columns)
        ],
        remainder='passthrough'
    )

    svm_classifier = SVC(kernel='linear')

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', svm_classifier)
    ])

    model.fit(X, y)

    print("Answer the following questions (yes/no):")

    user_responses = pd.DataFrame(columns=selected_columns)

    for question, column in zip(selected_columns, selected_columns):
        response = None
        while response not in ['yes', 'no']:
            response = input(question + "? ").lower()
            if response not in ['yes', 'no']:
                print("Please answer with yes or no.")

        user_responses[column] = [response]

    user_prediction = model.predict(user_responses)
    print("Predicted Disorder:", user_prediction[0])
