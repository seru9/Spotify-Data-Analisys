# Implementation Plan — Spotify Tracks Dataset
## Projekt PDU 2025/2026

> **Uwaga do kodu:** Fragmenty kodu w tym planie są wzorowane bezpośrednio na przykładach
> z wykładów PDU 2025/2026 (Anna Cena). Przy każdym bloku kodu wskazano numer wykładu,
> z którego pochodzi dana technika lub styl zapisu.

---

## Struktura projektu

```
projekt/
├── data/                  # pliki .csv z gatunkami (już gotowe)
├── scripts/
│   ├── 00_load_data.py
│   ├── 01_clean_data.py
│   ├── 02_analysis_popularity.py
│   ├── 03_analysis_genres.py
│   └── 04_kmeans_clustering.R
├── presentation/          # wygenerowane wykresy + slajdy
└── README.md
```

---

## Krok 0 — Wczytanie danych (`00_load_data.py`)

Iteracja po wszystkich plikach `.csv` w katalogu `data/`, które odpowiadają poszczególnym
gatunkom muzycznym. Każdy plik ma identyczną strukturę kolumn.

Techniki z wykładów:
- `glob.glob()` + pętla `for` po plikach — wzorzec omówiony na **wyk. 07** (automatyzacja przetwarzania plików)
- `os.path.join()` do budowania ścieżek w sposób niezależny od systemu operacyjnego — **wyk. 07**
- `os.path.exists()` do sprawdzenia czy katalog wyjściowy istnieje przed zapisem — **wyk. 07**
- `pd.read_csv()` wewnątrz pętli, analogicznie jak `X = pd.read_csv(plik)` na **wyk. 07**

```python
import pandas as pd
import glob
import os

# glob.glob() — wyk. 07: automatyzacja przetwarzania plików w katalogu
csv_files = glob.glob(os.path.join("data", "*.csv"))

dfs = []
for path in csv_files:
    df = pd.read_csv(path)   # wyk. 07: pd.read_csv w pętli po plikach
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Sprawdzamy katalog wyjściowy przed zapisem — wyk. 07: os.path.exists + os.mkdir
if not os.path.exists("data"):
    os.mkdir("data")

df_all.to_parquet("data/all_tracks.parquet", index=False)  # szybszy odczyt później
```

**Uwagi:**
- Zapisujemy scalony zbiór do `.parquet` — szybszy odczyt przy kolejnych uruchomieniach.
- Kolumna `track_genre` jest już obecna w każdym pliku, więc nie trzeba jej dodawać ręcznie.

---

## Krok 1 — Czyszczenie danych (`01_clean_data.py`)

| Problem | Działanie | Wykład |
|---|---|---|
| Diagnostyka wstępna | `df.info()` + `df.describe(include="all")` | wyk. 04 |
| Duplikaty `track_id` | `drop_duplicates()` — zachować wiersze gdy `track_genre` się różni, usunąć gdy cały wiersz identyczny | wyk. 05 |
| Brakujące wartości | Sprawdzić braki w `popularity`, `duration_ms`, cechach audio | wyk. 04 |
| `key == -1` | `replace(-1, pd.NA)` — brak danych, nie brak tonacji | wyk. 04 |
| `duration_ms` outliers | Filtrowanie przez maskę logiczną `df[warunek]` | wyk. 04 |
| `artists` — separator `;` | `str.split(";")` + `str.strip()` przy analizie per-artysta | wyk. 06 |

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("data/all_tracks.parquet")

# Diagnostyka przed czyszczeniem — wyk. 04: info() i describe() jako obowiązkowy
# krok wstępny przy każdej nowej ramce danych
df.info()
df.describe(include="all")

# Duplikaty — wyk. 05: drop_duplicates()
df = df.drop_duplicates()

# Tonacja: wartość -1 oznacza brak danych — wyk. 04: replace na NaN/pd.NA
df["key"] = df["key"].replace(-1, pd.NA)

# Długość: filtrowanie przez maskę logiczną — wyk. 04: df.loc[warunek, :]
df = df[(df["duration_ms"] > 10_000) & (df["duration_ms"] < 1_200_000)]

df.to_parquet("data/clean_tracks.parquet", index=False)
```

---

## Krok 2 — Analiza A: Popularność artystów i piosenek (`02_analysis_popularity.py`)

### Pytania badawcze

1. **Którzy artyści mają najwyższą medianową popularność?** (top 20)
2. **Jak rozkłada się popularność między gatunkami?** — box plot dla wybranych gatunków
3. **Czy cechy audio wpływają na popularność?** — heatmapa korelacji

### Wykresy

```
Wykres 1: Poziomy bar chart — top 20 artystów wg mediany popularity
Wykres 2: Box plot — popularity per genre (top 15 gatunków wg liczby utworów)
Wykres 3: Heatmapa korelacji — popularity vs cechy audio
```

Techniki z wykładów:
- `str.split(";")` + `str.strip()` do rozbicia kolumny `artists` — **wyk. 06** (przetwarzanie napisów, metody `.str`)
- `explode()` — rozwinięcie listy artystów na osobne wiersze
- `groupby(..., observed=True)` + `reset_index()` + `sort_values()` krok po kroku — **wyk. 05**
- `iloc[:20, :]` do wyboru top 20 po sortowaniu — **wyk. 04/05** (indeksowanie po pozycji)
- `seaborn.boxplot()` do rozkładu popularności — **wyk. 12** (wizualizacja; boxplot omówiony na przykładzie `iris`)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("data/clean_tracks.parquet")

# Rozbijamy kolumnę artists — str.split + str.strip, wyk. 06
# Analogia: w. 06 pokazuje w.str.strip().str.replace().str.split() na danych
df_artists = df.copy()
df_artists["artist"] = df_artists["artists"].str.split(";")
df_artists = df_artists.explode("artist")
df_artists["artist"] = df_artists["artist"].str.strip()

# Top 20 artystów — groupby + reset_index + sort_values, wyk. 05
# Analogia: wynik = tips.groupby('day')['total_bill'].mean()
#           wynik = wynik.reset_index()
#           wynik = wynik.sort_values('total_bill', ascending=False)
artist_pop = df_artists.groupby("artist", observed=True)["popularity"].median()
artist_pop = artist_pop.reset_index()
artist_pop = artist_pop.sort_values("popularity", ascending=False)
artist_pop = artist_pop.iloc[:20, :]  # wyk. 04: iloc po pozycji

# Wykres 2: box plot popularity per genre — wyk. 12 (boxplot na split danych)
top_genres = df["track_genre"].value_counts().head(15).index
df_top = df[df["track_genre"].isin(top_genres)]

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_top, x="track_genre", y="popularity", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig("presentation/wykres2_popularity_boxplot.png", dpi=150)

# Wykres 3: heatmapa korelacji
# Heatmapa korelacji pokazuje siłę i kierunek zależności liniowej między zmiennymi.
# Współczynnik korelacji Pearsona przyjmuje wartości od -1 do 1:
#   1.0  — idealna korelacja dodatnia (gdy jedna rośnie, druga też)
#  -1.0  — idealna korelacja ujemna (gdy jedna rośnie, druga maleje)
#   0.0  — brak zależności liniowej
audio_features = ["popularity", "danceability", "energy", "valence", "tempo", "acousticness"]
corr = df[audio_features].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
plt.tight_layout()
plt.savefig("presentation/wykres3_correlation_heatmap.png", dpi=150)
```

---

## Krok 3 — Analiza B: Gatunki muzyczne (`03_analysis_genres.py`)

### Pytania badawcze

1. **Które gatunki są "najweselsze"?** — średnia `valence` per genre
2. **Które gatunki mają najwięcej utworów z wulgarnymi treściami (`explicit`)?**
3. **Jaki jest "profil audio" każdego gatunku?** — radar chart dla wybranych cech

### Wykresy

```
Wykres 4: Poziomy bar chart — top 20 gatunków wg średniej valence
Wykres 5: Stacked bar / % bar — udział explicit per genre (top 20 gatunków)
Wykres 6: Radar chart — porównanie profilu audio 5–6 wybranych gatunków
```

Techniki z wykładów:
- `groupby(..., observed=True)` krok po kroku — **wyk. 05**
- `value_counts()` do zliczenia udziału `explicit` — **wyk. 05** (`tips.sex.value_counts()`)
- `reset_index()` po `groupby` — **wyk. 05** (konieczne do dalszego sortowania)
- `sort_values(..., ascending=False)` — **wyk. 05**
- `matplotlib` radar chart zbudowany ręcznie przez `np.linspace` i `ax.fill` — brak gotowej funkcji, robimy sami korzystając z wiedzy o tablicach numpy z **wyk. 02/03**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("data/clean_tracks.parquet")

audio_features = ["danceability", "energy", "acousticness",
                  "valence", "instrumentalness", "speechiness"]

# Które gatunki są najweselsze — groupby + reset_index + sort_values, wyk. 05
valence_by_genre = df.groupby("track_genre", observed=True)["valence"].mean()
valence_by_genre = valence_by_genre.reset_index()
valence_by_genre = valence_by_genre.sort_values("valence", ascending=False)

# Udział explicit per gatunek — value_counts, wyk. 05
# Analogia: tips.sex.value_counts() zlicza wystąpienia każdej kategorii
explicit_counts = df.groupby("track_genre", observed=True)["explicit"].value_counts()
explicit_counts = explicit_counts.reset_index()

# Profil audio per gatunek do radar chartu
genre_profile = df.groupby("track_genre", observed=True)[audio_features].mean()
genre_profile = genre_profile.reset_index()

# Radar chart — ręczna konstrukcja kąt po kącie, korzystamy z np.linspace (wyk. 02/03)
def radar_chart(genres, profile_df, features, filename):
    n = len(features)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # zamknięcie wykresu

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for genre in genres:
        row = profile_df[profile_df["track_genre"] == genre][features].values.flatten().tolist()
        row += row[:1]
        ax.plot(angles, row, label=genre)
        ax.fill(angles, row, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(filename, dpi=150)

selected = ["pop", "metal", "jazz", "classical", "hip-hop"]
radar_chart(selected, genre_profile, audio_features,
            "presentation/wykres6_radar_audio_profile.png")
```

---

## Krok 4 — Analiza C: Klasteryzacja gatunków w R (`04_kmeans_clustering.R`)

*(Zwiększa szansę na wysoką ocenę)*

### Pytanie badawcze
**Czy gatunki muzyczne naturalnie grupują się w klastry na podstawie cech audio?**
K-Means na średnich cechach audio per gatunek → czy np. metal/rock lądują razem, a folk/acoustic osobno?

### Dane wejściowe
Skrypt wczytuje CSV wyeksportowany z kroku 1 (patrz uwaga na końcu).

Techniki z wykładów:
- `read.csv()` do wczytania danych — **wyk. 11** (`bikes <- read.csv(files[1])`)
- `list.files(..., pattern="*.csv")` jako alternatywa do lokalizacji pliku — **wyk. 11**
- `dplyr::group_by()` + `summarise()` + `across()` do obliczenia średnich cech per gatunek — `dplyr` (część `tidyverse`)
- `dplyr::select()` do wyboru kolumn przed standaryzacją — `dplyr`
- `dplyr::mutate()` do dodania kolumn klastra i gatunku do ramki wynikowej — `dplyr`
- `kmeans()` + pętla `for` po zakresie k — **wyk. 13** (dokładnie ten wzorzec: `K <- 2:20; for (k in K) { model <- kmeans(X, centers = k) }`)
- `sapply()` do obliczenia WSS dla różnych k — **wyk. 09/11**
- `plot()` do wykresu metody łokcia — **wyk. 12** (`plot(x, type='b')`)
- scatter plot dwóch wybranych cech audio pokolorowany wg klastra — **wyk. 13** (`plot(X[,1], X[,2], col = model$cluster)`)
- `ggplot2` (`ggplot` + `geom_point` + `theme_minimal`) — rozszerza bazową wizualizację R z **wyk. 12**

> **Uwaga:** PCA (`prcomp()`) **nie było omówione na wykładach** — zastępujemy je prostszym podejściem: ręczny wybór dwóch cech audio (np. `energy` vs `valence`) jako osi X i Y scatter plota. Jest to uczciwe, czytelne i zgodne ze stylem z wyk. 13.

```r
library(tidyverse)   # zawiera dplyr, ggplot2, readr
library(arrow)       # odczyt parquet; alternatywnie read.csv() eksportu z Pythona

# Wczytanie — wyk. 11: read.csv() / list.files()
df <- read_parquet("data/clean_tracks.parquet")

# Cechy audio do klasteryzacji
audio_features <- c("danceability", "energy", "acousticness",
                    "valence", "instrumentalness", "speechiness", "tempo")

# Profil per gatunek — dplyr: group_by + summarise + across
genre_profile <- df %>%
  group_by(track_genre) %>%
  summarise(across(all_of(audio_features), mean, na.rm = TRUE)) %>%
  ungroup()

# Standaryzacja (K-Means wrażliwy na skalę) — dplyr: select do wyboru kolumn
genre_scaled <- genre_profile %>%
  select(all_of(audio_features)) %>%
  scale()

rownames(genre_scaled) <- genre_profile$track_genre
```

### Wybór liczby klastrów — metoda łokcia

```r
# sapply w pętli — wyk. 09/11
wss <- sapply(1:15, function(k) {
  kmeans(genre_scaled, centers = k, nstart = 25)$tot.withinss
})

# plot() — wyk. 12: plot(x, type='b') do wykresów liniowych
plot(1:15, wss, type = "b", pch = 19,
     xlab = "Liczba klastrów k", ylab = "WSS",
     main = "Metoda łokcia")
```

### K-Means + wizualizacja

```r
# kmeans + pętla for — wyk. 13: model <- kmeans(X, centers = k)
set.seed(42)
k <- 5  # dobrać po analizie łokcia
km <- kmeans(genre_scaled, centers = k, nstart = 25)

# Dodajemy wyniki klastrów do ramki danych — dplyr: mutate
genre_result <- genre_profile %>%
  mutate(cluster = factor(km$cluster))   # dplyr: mutate do dodania kolumn

# Scatter plot: energy vs valence, pokolorowany wg klastra
# Wzorzec z wyk. 13: plot(X[,1], X[,2], col = model$cluster)
# Tu używamy ggplot2 dla czytelniejszego wyniku z etykietami gatunków
ggplot(genre_result, aes(x = energy, y = valence, color = cluster, label = track_genre)) +
  geom_point(size = 3) +
  ggrepel::geom_text_repel(size = 2.5) +
  labs(title = "Klasteryzacja gatunków muzycznych (K-Means)",
       subtitle = paste0("k = ", k, " klastrów | osie: energy vs valence"),
       x = "energy (średnia per gatunek)",
       y = "valence (średnia per gatunek)") +
  theme_minimal()

ggsave("presentation/wykres7_kmeans_genres.png", width = 12, height = 8, dpi = 150)
```

```
Wykres 7: Scatter plot energy vs valence — gatunki pokolorowane wg klastra K-Means
Wykres 8: Metoda łokcia — dobór optymalnego k
```

**Uwaga:** jeśli `arrow` sprawia problemy, wystarczy w kroku 1 dodać eksport do CSV i wczytać przez `read.csv()` (wyk. 11):
```python
df.to_csv("data/clean_tracks.csv", index=False)
```
```r
df <- read.csv("data/clean_tracks.csv")  # wyk. 11
```

---

## Krok 5 — Prezentacja

- Format: **PDF lub HTML** (np. z `reveal.js` lub Quarto)
- Slajdy:
  1. Tytuł + autorzy
  2. Opis danych (skąd, ile rekordów, ile gatunków)
  3. Analiza A — popularność (2 slajdy)
  4. Analiza B — gatunki (2–3 slajdy)
  5. Analiza C — klasteryzacja (1–2 slajdy)
  6. Wnioski + źródło danych
- Cytowanie źródła (obowiązkowe wg treści zadania):

> Maharshi Pandya. (2022). *Spotify Tracks Dataset* [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/4372070

---

## Zestawienie technik wg wykładów

| Wykład | Technika | Zastosowanie w projekcie |
|---|---|---|
| wyk. 02/03 | `numpy`: tablice, `np.linspace`, wektoryzacja | Radar chart — kąty, obliczenia na tablicach |
| wyk. 04 | `df.info()`, `df.describe()`, `df.loc[warunek]`, `iloc` | Diagnostyka + filtrowanie w czyszczeniu |
| wyk. 05 | `groupby(observed=True)`, `reset_index()`, `sort_values()`, `value_counts()`, `drop_duplicates()` | Agregacje per artysta/gatunek, ranking |
| wyk. 06 | `.str.split()`, `.str.strip()`, `.str.replace(regex=True)` | Rozbijanie kolumny `artists` |
| wyk. 07 | `glob.glob()`, `os.path.join()`, `os.path.exists()`, `os.mkdir()` | Wczytywanie 114 plików CSV w pętli |
| wyk. 08–10 | Podstawy R: wektory, listy, ramki danych, `typeof`, `attributes` | Fundament skryptu R |
| wyk. 11 | `read.csv()`, `list.files()`, `merge()`, `sapply()` | Wczytanie danych + pomocnicze operacje w R |
| wyk. 11 + dplyr | `group_by()`, `summarise()`, `across()`, `select()`, `mutate()`, `ungroup()` | Agregacja cech audio per gatunek w R |
| wyk. 13 | `kmeans()`, pętla `for` po k, `plot(X[,1], X[,2], col=model$cluster)` | K-Means + scatter plot klastrów |
| wyk. 12 | `plot()`, `boxplot()`, `barplot()`, `hist()` (R); `sns.boxplot()` (Python) | Wizualizacje w obu językach |

---

## Zależności

### Python

```
pandas
numpy
matplotlib
seaborn
pyarrow           # obsługa parquet
```

```bash
pip install pandas numpy matplotlib seaborn pyarrow
```

### R

```r
install.packages(c("tidyverse", "arrow", "ggrepel"))
```

---

## Checklisty przed oddaniem

- [ ] Skrypty uruchamiają się od zera (od kroku 0)
- [ ] Prezentacja zawiera cytowanie źródła danych
- [ ] Archiwum `.zip` nazwane `nralbumu1_nralbumu2.zip`
- [ ] W archiwum: jeden katalog z prezentacją + wszystkie skrypty
- [ ] Wygłoszenie prezentacji (maks. 10 min dla 2 osób)
