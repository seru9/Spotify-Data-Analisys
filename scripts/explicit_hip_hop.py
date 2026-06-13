import pandas as pd

df = pd.read_parquet("data/clean_tracks.parquet")
# 1. Twój bazowy pivot (zmieniony na wszystkie gatunki, żeby niczego nie przegapić)
explicit_counts = df.groupby("track_genre", observed=True)["explicit"].value_counts().reset_index(name="count")
explicit_pivot = explicit_counts.pivot(index="track_genre", columns="explicit", values="count").fillna(0)

# 2. Filtrujemy tylko gatunki, które mają w nazwie "hip" lub "rap"
hiphop_explicit = explicit_pivot[explicit_pivot.index.str.contains("hip|rap", case=False)]

# 3. Dodajemy kolumnę z procentowym udziałem utworów explicit (%_explicit)
hiphop_explicit["total_tracks"] = hiphop_explicit[True] + hiphop_explicit[False]
hiphop_explicit["%_explicit"] = (hiphop_explicit[True] / hiphop_explicit["total_tracks"] * 100).round(2)

# 4. Sortujemy od najbardziej wulgarnych gatunków
hiphop_explicit = hiphop_explicit.sort_values(by="%_explicit", ascending=False)

print(hiphop_explicit)