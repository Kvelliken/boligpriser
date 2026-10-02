# Boligprisprognose

Prognoser for boligprisene i Norge og de største byene (Oslo med Bærum, Bergen,
Trondheim og Stavanger) 6 måneder, 1 år, 3 år og 5 år frem i tid, med
usikkerhetsspenn. Oppdateres automatisk hver måned med GitHub Actions og
publiseres med GitHub Pages.

## Kom i gang

1. Opprett et **offentlig** repo på GitHub og last opp hele innholdet med
   mappestrukturen intakt (se «Opplasting» under).
2. **Settings → Pages → Build and deployment → Source:** velg **GitHub Actions**.
3. **Actions → Oppdater data og prognose → Run workflow.** Første kjøring tar
   noen minutter. Når den er ferdig, ligger siden på
   `https://<brukernavn>.github.io/<repo>/`.

Workflowen kjører deretter av seg selv den 16. og 26. hver måned, og hver gang
du endrer filer i `src/`, `web/`, `config/` eller `tests/`.

### Opplasting

Mapper som starter med punktum (`.github`) er skjult i Finder og Utforsker og
blir lett ikke med. Den sikreste måten er GitHub Desktop eller git:

```bash
git clone https://github.com/<brukernavn>/<repo>.git
# kopier innholdet av zip-filen inn i mappen, deretter:
cd <repo>
git add .
git commit -m "Første versjon"
git push
```

Workflowen sjekker filstrukturen først og sier tydelig fra hvis en fil mangler
eller ligger i feil mappe.

## Struktur

```
config/sources.yaml          Datakilder, tabeller og filtre
src/boligprognose/
  ssb.py, market.py          Klienter for SSB (PxWebApi v2), Norges Bank og FRED
  filters.py                 Uttrekk av serier fra SSB-tabeller
  fetch.py                   Henter alle kilder -> data/raw
  panel.py                   Kvartals- og månedspaneler -> data/processed
  features.py                Forklaringsvariabler per region
  model.py                   Modeller, historisk test og usikkerhetsspenn
  forecast.py                Prognose -> output/forecast.json
  site.py                    Bygger nettsiden til _site/
  verify.py                  Kontrollerer kildene (egen workflow)
web/                         Nettsiden (HTML, CSS, JavaScript)
tests/                       Tester, inkludert ende-til-ende på syntetiske data
output/archive/              Alle tidligere prognoser, én fil per måned
data/vintages/               Datagrunnlaget slik det så ut ved hver kjøring
```

## Metode

Målet er realprisendringen (boligpris deflatert med KPI) fra siste kjente
kvartal. For hver horisont fra 1 til 24 kvartaler estimeres en egen modell
(«direkte» prognose), slik at forklaringsvariablene inngår med verdiene som er
kjent i dag og ikke trenger egne prognoser.

Ensemblet består av fire komponenter:

- **ridge_lang:** Ridge-regresjon på et panel av alle regionene, med
  variabler som har historikk fra 1990-tallet. Variablene er boliglånsrente etter
  skatt og inflasjon, renteendring, reell kredittvekst (K2), oljepris, prisnivå
  mot byggekostnader og mot resten av landet, og prisutviklingen siste kvartal og
  siste år.
- **ridge_utvidet:** Det samme pluss befolkningsvekst, igangsatte boliger per
  innbygger og vekst i husholdningenes realinntekt (fra ca. 2000).
- **drift:** Regionens historiske gjennomsnittlige realprisvekst.
- **uendret:** Ingen endring i realpris.

Vektene settes per horisont ut fra treffsikkerheten i en historisk test fra
2005, krympet halvveis mot like vekter for robusthet. Usikkerhetsspennene er
empiriske kvantiler av ensemblets faktiske bom i den samme testen, der modellen
på hvert tidspunkt bare så data som var kjent da. Spennene tvinges til ikke å
krympe med horisonten.

Boligprisene er kvartalsvise. Mellom kvartalstallene oppdateres prognosen hver
måned med ferskere tall for KPI, boliglånsrente og kreditt.

Nominelle priser beregnes fra realprisprognosen og en inflasjonsbane der dagens
tolvmånedersvekst går lineært mot 2 % over 24 måneder.

### Kjente begrensninger

- Den historiske testen bruker dagens reviderte data. Fra og med første kjøring
  lagres datagrunnlaget i `data/vintages/`, slik at en ekte sanntidstest blir
  mulig over tid.
- Usikkerhet i inflasjonen er ikke med i spennene for nominelle priser.
- Spennene for 3 og 5 år bygger på færre uavhengige observasjoner enn de korte
  horisontene og er derfor mer usikre i seg selv.

## Lokalt

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest -q tests
python -m boligprognose.fetch
python -m boligprognose.panel
python -m boligprognose.forecast
python -m boligprognose.site
cd _site && python -m http.server 8000   # åpne http://localhost:8000
```

## Kilder og lisenser

- **Statistisk sentralbyrå** (PxWebApi v2), CC BY 4.0. Kilde må oppgis.
- **Norges Bank**, åpne data via SDMX-API.
- **FRED / U.S. EIA**: Brent-oljepris (DCOILBRENTEU).

Prognosene er modellberegninger og ikke råd om kjøp eller salg av bolig.
