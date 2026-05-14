import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# =========================
# CZESC 0 - GENEROWANIE DANYCH
# =========================

np.random.seed(42)

n = 500
klienci = [
    "Anna Kowalska", "  Jan Nowak", "Anna Kowalska", "PIOTR WIŚNIEWSKI",
    "katarzyna lewandowska", "Tomasz Zieliński ", "Marta Wójcik",
    "anna kowalska ", "Krzysztof Kamiński", " Magdalena Dąbrowska"
]
produkty = [
    "Laptop", "Mysz", "Klawiatura", "Monitor", "laptop", "MYSZ",
    "Słuchawki", "Pendrive", "monitor", "Webcam"
]
kategorie = [
    "Elektronika", "elektronika", "ELEKTRONIKA", "Akcesoria",
    "akcesoria", "Akcesoria "
]
miasta = [
    "Warszawa", "Kraków", "warszawa", "Gdańsk", "WROCŁAW",
    "Poznań", "Łódź ", " Warszawa", "kraków"
]

start_date = datetime(2025, 1, 1)

daty_iso = [
    (start_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
    for d in np.random.randint(0, 300, n // 2)
]

daty_pl = [
    (start_date + timedelta(days=int(d))).strftime("%d.%m.%Y")
    for d in np.random.randint(0, 300, n // 2)
]

daty = daty_iso + daty_pl
np.random.shuffle(daty)

df = pd.DataFrame({
    "order_id": range(1001, 1001 + n),
    "klient": np.random.choice(klienci, n),
    "produkt": np.random.choice(produkty, n),
    "kategoria": np.random.choice(kategorie, n),
    "miasto": np.random.choice(miasta, n),
    "ilosc": np.random.choice(
        [1, 2, 3, 5, -1, 0],
        n,
        p=[0.5, 0.2, 0.15, 0.1, 0.025, 0.025]
    ),
    "cena_jednostkowa": np.random.choice(
        ["199.99", "299,99", "1 499.00", "89.50", "2999", "399.00 zł", None, "abc"],
        n
    ),
    "data_zamowienia": daty,
    "email": np.random.choice(
        [
            "anna@gmail.com", "JAN@WP.PL", "piotr.w@onet", "marta@gmail.com",
            "tomasz@interia.pl", None, "krzysztof.k@gmail.com", "brak"
        ],
        n
    )
})

for col in ["miasto", "kategoria", "data_zamowienia"]:
    df.loc[df.sample(frac=0.05, random_state=1).index, col] = np.nan

df = pd.concat([df, df.sample(20, random_state=2)], ignore_index=True)

df.to_csv("zamowienia_messy.csv", index=False)
print(f"Wygenerowano plik zamowienia_messy.csv — {len(df)} wierszy")


# =========================
# CZESC 1 - EKSPLORACJA
# =========================

df = pd.read_csv("zamowienia_messy.csv")

print("\n=== KSZTALT DANYCH ===")
print(df.shape)

print("\n=== INFO ===")
print(df.info())

print("\n=== OPIS DANYCH ===")
print(df.describe(include="all"))

print("\n=== BRAKI DANYCH ===")
print(df.isnull().sum())

print("\n=== DUPLIKATY ===")
print(df.duplicated().sum())

print("\n=== VALUE COUNTS ===")
for col in ["klient", "produkt", "kategoria", "miasto", "email"]:
    print(f"\nKolumna: {col}")
    print(df[col].value_counts(dropna=False))


# Problemy z jakoscia danych:
# 1. Wystepuja duplikaty wierszy.
# 2. W kolumnach tekstowych sa spacje na poczatku lub koncu.
# 3. Te same wartosci zapisane sa rozna wielkoscia liter, np. Laptop/laptop.
# 4. Wystepuja braki danych w kolumnach miasto, kategoria, data_zamowienia i email.
# 5. Kolumna cena_jednostkowa zawiera tekst, przecinki, spacje, "zł", None oraz wartosc "abc".
# 6. Kolumna data_zamowienia ma dwa formaty dat.
# 7. Kolumna ilosc zawiera wartosci bledne: 0 oraz -1.
# 8. Niektore emaile sa niepoprawne, np. "brak" albo "piotr.w@onet".


# =========================
# CZESC 2 - CZYSZCZENIE
# =========================

df = df.drop_duplicates()

df["klient"] = df["klient"].str.strip().str.title()
df["produkt"] = df["produkt"].str.strip().str.title()
df["kategoria"] = df["kategoria"].str.strip().str.lower()
df["miasto"] = df["miasto"].str.strip().str.title()

df["data_zamowienia"] = pd.to_datetime(
    df["data_zamowienia"],
    errors="coerce",
    format="mixed"
)

df["cena_jednostkowa"] = (
    df["cena_jednostkowa"]
    .astype(str)
    .str.replace("zł", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.replace(",", ".", regex=False)
)

df["cena_jednostkowa"] = pd.to_numeric(
    df["cena_jednostkowa"],
    errors="coerce"
)

df["miasto"] = df["miasto"].fillna("unknown")
df["kategoria"] = df["kategoria"].fillna("unknown")
df["email"] = df["email"].fillna("brak_emaila")

df = df.dropna(subset=["cena_jednostkowa", "data_zamowienia"])

df = df[df["ilosc"] > 0]


# =========================
# CZESC 3 - TRANSFORMACJE
# =========================

df["wartosc_zamowienia"] = df["ilosc"] * df["cena_jednostkowa"]

df["rok"] = df["data_zamowienia"].dt.year
df["miesiac"] = df["data_zamowienia"].dt.month
df["nazwa_dnia"] = df["data_zamowienia"].dt.day_name()

df["email_poprawny"] = df["email"].str.match(
    r"^[\w\.-]+@[\w\.-]+\.\w+$",
    na=False
)


# =========================
# CZESC 4 - ANALIZA
# =========================

print("\n=== LACZNA WARTOSC ZAMOWIEN W KAZDYM MIESIACU ===")
wartosc_miesiac = (
    df.groupby("miesiac")["wartosc_zamowienia"]
    .sum()
    .sort_index()
)
print(wartosc_miesiac)

print("\n=== TOP 5 KLIENTOW ===")
top_klienci = (
    df.groupby("klient")["wartosc_zamowienia"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print(top_klienci)

print("\n=== SREDNIA WARTOSC ZAMOWIENIA W KATEGORII ===")
srednia_kategoria = (
    df.groupby("kategoria")["wartosc_zamowienia"]
    .mean()
    .sort_values(ascending=False)
)
print(srednia_kategoria)


# =========================
# CZESC 5 - WIZUALIZACJA
# =========================

wartosc_miesiac.plot(kind="bar")
plt.title("Laczna wartosc zamowien w kazdym miesiacu")
plt.xlabel("Miesiac")
plt.ylabel("Wartosc zamowien")
plt.tight_layout()
plt.savefig("wartosc_zamowien_miesiac.png")
plt.show()


# =========================
# CZESC 6 - ZAPIS
# =========================

df.to_csv("zamowienia_clean.csv", index=False)

print("\nZapisano oczyszczone dane do pliku zamowienia_clean.csv")
print("Zapisano wykres do pliku wartosc_zamowien_miesiac.png")
print(f"Liczba wierszy po czyszczeniu: {len(df)}")