TARGET = "churn"

X = df.drop(columns=[TARGET])

y = df[TARGET]

numeric_features = [

    "monthly_charges",

    "total_charges",

    "tenure_months",

    "support_tickets",

    "complaint_count",

    "avg_monthly_usage_gb"

]

categorical_features = [

    "gender",

    "city",

    "contract_type",

    "subscription_type",

    "internet_service"

]

y = y.map({

    "No":0,

    "Yes":1

})

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (

    OneHotEncoder,

    StandardScaler

)

numeric_pipeline = Pipeline([

    ("scaler", StandardScaler())

])

categorical_pipeline = Pipeline([

    ("encoder",

     OneHotEncoder(

         handle_unknown="ignore"

     ))

])

preprocessor = ColumnTransformer(

    [

        (

            "num",

            numeric_pipeline,

            numeric_features

        ),

        (

            "cat",

            categorical_pipeline,

            categorical_features

        )

    ]

)

preprocessor.fit(X_train)

X_train = preprocessor.transform(X_train)

X_test = preprocessor.transform(X_test)

import os
import joblib

os.makedirs('models', exist_ok=True)

joblib.dump(

    preprocessor,

    "models/preprocessor.pkl"

)

