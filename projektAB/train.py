

import ast
import os
import re
import warnings

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


warnings.filterwarnings("ignore")


DATA_PATH = "data/books.csv"
MODEL_PATH = "models/best_model.joblib"


ALLOWED_GENRES = [
    "Fantasy",
    "Science Fiction",
    "Mystery",
    "Thriller",
    "Romance",
    "Horror",
    "Historical Fiction",
    "Adventure"
]


GENRE_KEYWORDS = {
    "Science Fiction": [
        "science fiction", "sci fi", "sci-fi", "space", "spaceship",
        "spaceships", "robot", "robots", "artificial intelligence", "ai",
        "future", "galaxy", "planet", "planets", "mars", "alien",
        "aliens", "technology", "cyberpunk", "dystopia", "time travel"
    ],

    "Fantasy": [
        "fantasy", "magic", "magical", "dragon", "dragons", "wizard",
        "wizards", "sorcerer", "sorcerers", "witch", "witches", "spell",
        "spells", "kingdom", "sword", "prophecy", "ancient magic",
        "dark sorcerer", "elf", "elves", "fairy", "fairies"
    ],

    "Mystery": [
        "mystery", "detective", "crime", "murder", "investigation",
        "clue", "clues", "suspect", "case", "disappearance",
        "locked room", "secret", "evidence"
    ],

    "Thriller": [
        "thriller", "suspense", "conspiracy", "dangerous", "escape",
        "killer", "chase", "psychological thriller", "spy", "agent",
        "terror", "threat", "assassin", "following her", "want her dead",
        "serial killer", "race against time"
    ],

    "Romance": [
        "romance", "love", "relationship", "fall in love", "romantic",
        "marriage", "heartbreak", "passion", "boyfriend", "girlfriend",
        "wedding", "kiss"
    ],

    "Horror": [
        "horror", "ghost", "ghosts", "haunted", "haunted house",
        "haunted forest", "evil", "evil spirit", "demon", "nightmare",
        "terrifying", "dark house", "abandoned house", "monster", "blood",
        "scream", "screams", "fear", "creature", "creature hunts",
        "possession", "possessed", "curse", "cursed", "trapped",
        "survive the night", "one by one", "whispering", "empty room"
    ],

    "Historical Fiction": [
        "historical fiction", "historical", "world war", "war",
        "soldier", "nurse", "empire", "century", "king", "queen",
        "victorian", "medieval", "ancient", "revolution"
    ],

    "Adventure": [
        "adventure", "quest", "journey", "exploration", "treasure",
        "island", "jungle", "expedition", "explorer", "survival",
        "voyage", "mountain", "dangerous journey", "hidden treasure",
        "mysterious island", "cross mountains"
    ]
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_genres(value):
    if pd.isna(value):
        return []

    value = str(value)

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            genres = [str(g).strip().lower() for g in parsed]
        else:
            genres = [str(parsed).strip().lower()]

    except Exception:
        genres = re.split(r"[,;|/]", value)
        genres = [g.strip().lower() for g in genres if g.strip()]

    ignored_genres = {
        "science fiction fantasy",
        "audiobook",
        "audio",
        "fiction",
        "adult",
        "novels",
        "novel",
        "books",
        "literature",
        "young adult",
        "classics",
        "default",
        "book club"
    }

    cleaned_genres = []

    for genre in genres:
        if genre not in ignored_genres:
            cleaned_genres.append(genre)

    return cleaned_genres


def map_to_main_genre(genres, description):
    genre_text_list = [g.lower().strip() for g in genres]
    description_text = str(description).lower()

    scores = {genre: 0 for genre in ALLOWED_GENRES}

    genre_aliases = {
        "Science Fiction": ["science fiction", "sci fi", "sci-fi", "space", "cyberpunk", "dystopia"],
        "Fantasy": ["fantasy", "magic", "dragons", "dragon", "wizards", "wizard", "witches", "witch", "paranormal", "supernatural" ],
        "Mystery": ["mystery", "crime", "detective"],
        "Thriller": ["thriller", "suspense" ],
        "Romance": ["romance", "love"],
        "Horror": ["horror", "ghosts", "ghost", "haunted"],
        "Historical Fiction": [ "historical fiction", "historical"],
        "Adventure": ["adventure"]
    }


    for main_genre, aliases in genre_aliases.items():
        for genre in genre_text_list:
            for alias in aliases:
                if genre == alias:
                    scores[main_genre] += 10


    for main_genre, keywords in GENRE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in description_text:
                scores[main_genre] += 1

    best_genre = max(scores, key=scores.get)

    if scores[best_genre] == 0:
        return None

    return best_genre


def load_and_prepare_data():
    print("Ucitavanje CSV datoteke...")

    df = pd.read_csv(DATA_PATH)

    print(f"Ukupan broj redaka u CSV-u: {len(df)}")
    print(f"Stupci u datasetu: {list(df.columns)}")

    if "description" not in df.columns:
        raise ValueError("CSV nema stupac 'description'.")

    if "genres" not in df.columns:
        raise ValueError("CSV nema stupac 'genres'.")

    useful_columns = ["description", "genres"]

    if "title" in df.columns:
        useful_columns.append("title")

    df = df[useful_columns].copy()

    df = df.dropna(subset=["description", "genres"])

    if "title" in df.columns:
        df["title"] = df["title"].fillna("")
        df["full_text"] = df["title"].astype(str) + " " + df["description"].astype(str)
    else:
        df["full_text"] = df["description"].astype(str)

    df["full_text"] = df["full_text"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)

    df["genre_list"] = df["genres"].apply(parse_genres)

    df["genre"] = df.apply(
        lambda row: map_to_main_genre(row["genre_list"], row["description"]),
        axis=1
    )

    df = df.dropna(subset=["genre"])
    df = df[df["genre"].isin(ALLOWED_GENRES)]
    df = df[df["full_text"].str.len() > 50]

    print("\nBroj primjera po zanru:")
    print(df["genre"].value_counts())

    print(f"\nUkupan broj koristenih zapisa: {len(df)}")
    print(f"Broj koristenih zanrova: {df['genre'].nunique()}")

    if df["genre"].nunique() < 2:
        raise ValueError("Nema dovoljno zanrova za treniranje modela.")

    return df


def balance_training_data(X_train, y_train):
    """
    Umjereno balansiranje.
    Slabije klase se povecaju do medijana, ali se ne forsira samo jedan zanr.
    """

    train_df = pd.DataFrame({
        "text": X_train,
        "genre": y_train
    })

    counts = train_df["genre"].value_counts()
    target_count = int(counts.median())

    balanced_parts = []

    for genre, group in train_df.groupby("genre"):
        if len(group) < target_count:
            sampled = group.sample(
                n=target_count,
                replace=True,
                random_state=42
            )
        else:
            sampled = group.sample(
                n=target_count,
                replace=False,
                random_state=42
            )

        balanced_parts.append(sampled)

    balanced_df = pd.concat(balanced_parts)
    balanced_df = balanced_df.sample(frac=1, random_state=42)

    print("\nRaspodjela zanrova nakon balansiranja trening skupa:")
    print(balanced_df["genre"].value_counts())

    return balanced_df["text"], balanced_df["genre"]


def add_manual_training_examples(X_train, y_train):
    """
    Dodajemo malo dodatnih trening primjera za slabe zanrove.
    Ovo se dodaje samo u trening, ne u test skup.
    """

    manual_examples = [
        # Horror
        ("A family moves into an abandoned house where they hear voices, see ghosts, and discover an evil demon.", "Horror"),
        ("A group of friends enters a haunted forest where a terrifying creature hunts them during the night.", "Horror"),
        ("A child hears whispering from an empty room and realizes a ghost is living inside the house.", "Horror"),
        ("People trapped in a cursed village are attacked by monsters after midnight.", "Horror"),
        ("A haunted house slowly drives a family insane as ghosts appear in every room.", "Horror"),

        # Thriller
        ("A journalist uncovers a dangerous conspiracy and realizes that powerful people want her dead.", "Thriller"),
        ("A man discovers someone is following him and must escape before the killer finds him.", "Thriller"),
        ("A spy is hunted by assassins after stealing secret government documents.", "Thriller"),
        ("A woman receives threatening messages and realizes the murderer knows where she lives.", "Thriller"),
        ("A detective races against time to stop a serial killer before another victim disappears.", "Thriller"),

        # Adventure
        ("A group of explorers travels through dangerous jungles searching for a hidden treasure.", "Adventure"),
        ("Three strangers cross mountains, rivers, and unknown lands during a dangerous journey.", "Adventure"),
        ("A young explorer sails to a mysterious island in search of a lost civilization.", "Adventure"),
        ("A team begins an expedition across the desert to find an ancient temple.", "Adventure"),
        ("A group of travelers must survive on a remote island after their ship is destroyed.", "Adventure"),
    ]

    manual_df = pd.DataFrame(manual_examples, columns=["text", "genre"])

    X_train_extended = pd.concat(
        [pd.Series(X_train), manual_df["text"]],
        ignore_index=True
    )

    y_train_extended = pd.concat(
        [pd.Series(y_train), manual_df["genre"]],
        ignore_index=True
    )

    print("\nDodani dodatni trening primjeri:")
    print(manual_df["genre"].value_counts())

    return X_train_extended, y_train_extended


def build_models():
    tfidf_strong = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True
    )

    tfidf_fast = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf_strong),
            ("classifier", LogisticRegression(
                max_iter=3000,
                C=4.0,
                class_weight="balanced",
                n_jobs=-1
            ))
        ]),

        "Linear SVM": Pipeline([
            ("tfidf", tfidf_strong),
            ("classifier", LinearSVC(
                C=1.5,
                class_weight="balanced"
            ))
        ]),

        "Complement Naive Bayes": Pipeline([
            ("tfidf", tfidf_strong),
            ("classifier", ComplementNB(alpha=0.3))
        ]),

        "Random Forest": Pipeline([
            ("tfidf", tfidf_fast),
            ("classifier", RandomForestClassifier(
                n_estimators=80,
                max_depth=40,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])
    }

    return models


def train_and_evaluate():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = load_and_prepare_data()

    X = df["full_text"]
    y = df["genre"]

    print("\nDijelim podatke na 80% trening i 20% test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Broj trening primjera prije balansiranja: {len(X_train)}")
    print(f"Broj test primjera: {len(X_test)}")

    print("\nRaspodjela zanrova u trening skupu prije balansiranja:")
    print(y_train.value_counts())

    print("\nRaspodjela zanrova u test skupu:")
    print(y_test.value_counts())

    train_export = pd.DataFrame({
        "description": X_train,
        "genre": y_train
    })

    test_export = pd.DataFrame({
        "description": X_test,
        "genre": y_test
    })

    train_export.to_csv("reports/train_data.csv", index=False)
    test_export.to_csv("reports/test_data.csv", index=False)

    X_train_balanced, y_train_balanced = balance_training_data(X_train, y_train)

    X_train_balanced, y_train_balanced = add_manual_training_examples(
        X_train_balanced,
        y_train_balanced
    )

    models = build_models()

    results = []

    best_model = None
    best_model_name = None
    best_macro_f1 = -1
    best_predictions = None

    for model_name, model in models.items():
        print("\n----------------------------------------")
        print(f"Treniram model: {model_name}")

        model.fit(X_train_balanced, y_train_balanced)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro")
        weighted_f1 = f1_score(y_test, predictions, average="weighted")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1
        })

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model = model
            best_model_name = model_name
            best_predictions = predictions

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Macro F1", ascending=False)

    results_df.to_csv("reports/model_results.csv", index=False)

    report = classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )

    with open("reports/classification_report.txt", "w", encoding="utf-8") as file:
        file.write(f"Najbolji model prema Macro F1: {best_model_name}\n")
        file.write(f"Najbolji Macro F1: {best_macro_f1:.4f}\n\n")
        file.write(report)

    cm = confusion_matrix(
        y_test,
        best_predictions,
        labels=best_model.classes_
    )

    cm_df = pd.DataFrame(
        cm,
        index=best_model.classes_,
        columns=best_model.classes_
    )

    cm_df.to_csv("reports/confusion_matrix.csv")

    # Za aplikaciju spremamo Logistic Regression jer se pokazao stabilnijim u rucnom testiranju
    # i podrzava prikaz postotaka pomocu predict_proba.
    
    final_model_name = "Logistic Regression"
    final_model = models[final_model_name]
    final_model.fit(X_train_balanced, y_train_balanced)

    joblib.dump(final_model, MODEL_PATH)

    print("\n========================================")
    print("Treniranje zavrseno.")
    print(f"Najbolji model prema Macro F1: {best_model_name}")
    print(f"Najbolji Macro F1: {best_macro_f1:.4f}")
    print(f"Model spremljen za aplikaciju: {final_model_name}")
    print(f"Model spremljen u: {MODEL_PATH}")
    print("Rezultati spremljeni u folder: reports/")
    print("Train podaci spremljeni u: reports/train_data.csv")
    print("Test podaci spremljeni u: reports/test_data.csv")
    print("========================================")


if __name__ == "__main__":
    train_and_evaluate()