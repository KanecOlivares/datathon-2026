"""Feature engineering utilities."""

from __future__ import annotations

import pandas as pd

_YES_NO_MAP = {"No": 0, "Yes": 1}
_THREE_LEVEL_MAP = {"Low": 1, "Medium": 2, "High": 3}
_PEER_INFLUENCE_MAP = {"Negative": -1, "Neutral": 0, "Positive": 1}
_PARENT_EDUCATION_MAP = {"High School": 1, "College": 2, "Postgraduate": 3}
_DISTANCE_MAP = {"Far": 1, "Moderate": 2, "Near": 3}
_SCHOOL_TYPE_MAP = {"Public": 0, "Private": 1}
_GENDER_MAP = {"Female": 0, "Male": 1}


def _add_encoded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create lightweight numeric encodings for ordered or binary categories."""
    featured = df.copy()
    featured["Parental_Involvement_Encoded"] = featured["Parental_Involvement"].map(_THREE_LEVEL_MAP)
    featured["Access_to_Resources_Encoded"] = featured["Access_to_Resources"].map(_THREE_LEVEL_MAP)
    featured["Extracurricular_Activities_Encoded"] = featured["Extracurricular_Activities"].map(_YES_NO_MAP)
    featured["Motivation_Level_Encoded"] = featured["Motivation_Level"].map(_THREE_LEVEL_MAP)
    featured["Internet_Access_Encoded"] = featured["Internet_Access"].map(_YES_NO_MAP)
    featured["Family_Income_Encoded"] = featured["Family_Income"].map(_THREE_LEVEL_MAP)
    featured["Teacher_Quality_Encoded"] = featured["Teacher_Quality"].map(_THREE_LEVEL_MAP)
    featured["School_Type_Encoded"] = featured["School_Type"].map(_SCHOOL_TYPE_MAP)
    featured["Peer_Influence_Encoded"] = featured["Peer_Influence"].map(_PEER_INFLUENCE_MAP)
    featured["Learning_Disabilities_Encoded"] = featured["Learning_Disabilities"].map(_YES_NO_MAP)
    featured["Parental_Education_Level_Encoded"] = featured["Parental_Education_Level"].map(_PARENT_EDUCATION_MAP)
    featured["Distance_from_Home_Encoded"] = featured["Distance_from_Home"].map(_DISTANCE_MAP)
    featured["Gender_Encoded"] = featured["Gender"].map(_GENDER_MAP)
    return featured


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Build compact interaction features that are plausible drivers of exam score."""
    featured = df.copy()
    featured["Study_Attendance_Product"] = featured["Hours_Studied"] * featured["Attendance"]
    featured["Study_Sleep_Balance"] = featured["Hours_Studied"] / featured["Sleep_Hours"].clip(lower=1)
    featured["Academic_Support_Index"] = (
        featured["Access_to_Resources_Encoded"]
        + featured["Teacher_Quality_Encoded"]
        + featured["Internet_Access_Encoded"]
        + featured["Tutoring_Sessions"]
    )
    featured["Home_Support_Index"] = (
        featured["Parental_Involvement_Encoded"]
        + featured["Parental_Education_Level_Encoded"]
        + featured["Family_Income_Encoded"]
    )
    featured["Wellbeing_Index"] = featured["Sleep_Hours"] + featured["Physical_Activity"]
    featured["Previous_Score_Gap"] = featured["Exam_Score"] - featured["Previous_Scores"]
    return featured


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add encoded and derived features to the cleaned dataset."""
    featured = _add_encoded_columns(df)
    featured = _add_derived_columns(featured)
    return featured
