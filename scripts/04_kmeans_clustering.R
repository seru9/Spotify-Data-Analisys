# 04_kmeans_clustering.R
# Analiza C: Grupowanie gatunków muzycznych metodą K-Średnich (K-Means)
# Uruchomienie: Rscript scripts/04_kmeans_clustering.R

library(dplyr)  # Używamy TYLKO dplyr do manipulacji danymi

# ------------------------------------------------------------------
# Wczytanie danych
# ------------------------------------------------------------------
if (file.exists("../data/clean_tracks.csv")) {
  df <- read.csv("../data/clean_tracks.csv")
  cat("Wczytano: ../data/clean_tracks.csv\n")
} else {
  stop("Brak pliku danych! Uruchom najpierw 01_clean_data.py i dodaj eksport CSV.")
}

cat(sprintf("Wymiar danych: %d wierszy, %d kolumn\n", nrow(df), ncol(df)))

# ------------------------------------------------------------------
# Cechy audio do grupowania
# ------------------------------------------------------------------
audio_features <- c("danceability", "energy", "acousticness",
                    "valence", "instrumentalness", "speechiness", "tempo")

# ------------------------------------------------------------------
# Profil per gatunek — dplyr: group_by + summarise + across
# ------------------------------------------------------------------
genre_profile <- df %>%
  group_by(track_genre) %>%
  summarise(across(all_of(audio_features), mean, na.rm = TRUE)) %>%
  ungroup()

cat(sprintf("Liczba gatunków: %d\n", nrow(genre_profile)))

# ------------------------------------------------------------------
# Standaryzacja (K-Means wrażliwy na skalę)
# ------------------------------------------------------------------
genre_scaled <- genre_profile %>%
  select(all_of(audio_features)) %>%
  scale()

rownames(genre_scaled) <- genre_profile$track_genre

# ------------------------------------------------------------------
# Dobór optymalnego k (metoda łokcia)
# ------------------------------------------------------------------
cat("Obliczanie WSS dla k = 1..15...\n")

wss <- sapply(1:15, function(k) {
  kmeans(genre_scaled, centers = k, nstart = 25)$tot.withinss
})

png("../presentation/wykres8_elbow_method.png", width = 800, height = 600)
# Bazowy plot liniowy
plot(1:15, wss, type = "b", pch = 19, col = "steelblue",
     xlab = "Liczba grup (k)", ylab = "WSS (Within-cluster Sum of Squares)",
     main = "Metoda łokcia — dobór optymalnej liczby grup")
dev.off()
cat("Zapisano: wykres8_elbow_method.png\n")

# ------------------------------------------------------------------
# K-Means dla wybranego k
# ------------------------------------------------------------------
set.seed(42)
k <- 5  # dobrano na podstawie metody łokcia
km <- kmeans(genre_scaled, centers = k, nstart = 25)

cat(sprintf("K-Means z k=%d — rozmiary grup:\n", k))
print(table(km$cluster))

# Dodajemy wyniki grup do ramki danych za pomocą dplyr
genre_result <- genre_profile %>%
  mutate(cluster = factor(km$cluster))

# ------------------------------------------------------------------
# Wielopanelowy wykres scatter plot (Siatka 2x2) za pomocą bazowego plot()
# ------------------------------------------------------------------
png("../presentation/wykres7_kmeans_genres_grid.png", width = 1400, height = 1200)

# Ustawienie siatki wykresów: 2 wiersze, 2 kolumny oraz marginesów
par(mfrow = c(2, 2), mar = c(5, 5, 4, 2))

# Generujemy paletę bogatą w kolory (tęcza), ponieważ domyślne palette() ma tylko 8 kolorów, a k=12
kolory_grup <- rainbow(k)

# Uniwersalna funkcja pomocnicza do renderowania paneli
rysuj_panel <- function(cecha_x, cecha_y, nazwa_x, nazwa_y) {
  # Rysujemy punkty (kolorowane na podstawie przypisanej grupy)
  plot(genre_result[[cecha_x]], genre_result[[cecha_y]],
       col = kolory_grup[genre_result$cluster],
       pch = 19, 
       cex = 1.6,
       xlab = nazwa_x,
       ylab = nazwa_y,
       main = paste(nazwa_y, "vs", nazwa_x))
  
  grid()
  
  # Dodanie etykiet tekstowych (nazw gatunków) lekko nad punktami
  text(genre_result[[cecha_x]], genre_result[[cecha_y]], 
       labels = genre_result$track_genre, 
       pos = 3, 
       cex = 0.7, 
       col = "gray25")
}

# Panel 1: Energia vs Taneczność (Rozróżnienie dynamiki imprezowej i agresywnej)
rysuj_panel("energy", "danceability", "energy (średnia)", "danceability (średnia)")

# Legenda umieszczona tylko na pierwszym wykresie w dogodnym miejscu, aby nie powielać jej 4 razy
legend("topleft", 
       legend = paste("Grupa", 1:k), 
       col = kolory_grup, 
       pch = 19, 
       bty = "n",
       cex = 0.85,
       title = "Wyznaczone grupy")

# Panel 2: Akustyczność vs Instrumentalność (Wyodrębnienie muzyki klasycznej/ambientu)
rysuj_panel("acousticness", "instrumentalness", "acousticness", "instrumentalness")

# Panel 3: Energia vs Wesołość (Klasyczny podział na emocjonalny klimat utworów)
rysuj_panel("energy", "valence", "energy", "valence")

# Panel 4: Tempo vs Gadatliwość (Kluczowe dla wyodrębnienia Rapu/Hip-Hopu i Podcastów)
rysuj_panel("tempo", "speechiness", "tempo", "speechiness")

dev.off()
cat("Zapisano: wykres7_kmeans_genres_grid.png\n")

# ------------------------------------------------------------------
# Gatunki w poszczególnych grupach
# ------------------------------------------------------------------
cat("\n=== Gatunki w poszczególnych grupach ===\n")
for (i in 1:k) {
  cat(sprintf("\nGrupa %d:\n", i))
  gatunki <- genre_result %>%
    filter(cluster == i) %>%
    pull(track_genre)
  cat(paste(gatunki, collapse = ", "), "\n")
}