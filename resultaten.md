# Resultaten: smelt langs de K-transect

Dit document vat samen wat er met de stationsdata in `data/` is gedaan en wat daaruit kwam. Het gaat om drie delen:

1. **Smelt per station**: een vergelijking van de berekende smelt tussen alle stations ([code/melt_comparison.py](code/melt_comparison.py)).
2. **Hoogte en albedo**: hoe de energiebalans en de smelt veranderen met de hoogte, en welke rol het albedo speelt ([code/elevation_albedo.py](code/elevation_albedo.py)).
3. **Foutmarges en correlaties**: hoe zeker de uitkomsten zijn, en hoe de correlaties zich verhouden tot Kuipers Munneke et al. (2018) ([code/uncertainty.py](code/uncertainty.py)).

Alle figuren en tabellen staan in [output/](output/).

---

## 1. Smelt per station

### Wat er gedaan is

Voor elk station is de smelt berekend uit de energiebalans van het oppervlak:

> **M = netto kortgolvig + netto langgolvig + voelbare warmte (SHF) + latente warmte (LHF)**

Er wordt alleen smelt geteld als het oppervlak op 0 °C is en er energie over is. De warmtestroom in het ijs is weggelaten. De smeltenergie is omgerekend naar mm waterequivalent (mm w.e.) en daarna opgeteld per uur, dag, week, maand of jaar. Een periode krijgt alleen een waarde als er minstens 80% data is.

| Stations | Hoogte | Straling | Turbulente fluxen (SHF, LHF) |
|---|---|---|---|
| AWS5, AWS6, AWS9, AWS10 | 500 / 1000 / 1500 / 1850 m | gemeten | berekend met de bulkmethode uit wind, temperatuur, vochtigheid en oppervlaktetemperatuur |
| SHR (bij KAN_L) | niet in data | van AWS5 (benadering) | SHF gemeten (eddy covariance), LHF zelf berekend |

De eddy-covariancemast S5 staat op dezelfde plek als AWS5 en is daarom niet als apart station meegenomen.

**Ruwheid.** De ruwheidslengtes volgen zo goed mogelijk Kuipers Munneke et al. (2018):

| Station | Ruwheid voor wind (z0m) | Ruwheid voor warmte en vocht (zh, zq) |
|---|---|---|
| AWS5 | gemeten door de eddy-covariancemast S5, lopend gemiddelde over 20 dagen; jaren zonder mastdata krijgen het seizoensgemiddelde (~4 mm in april tot ~33 mm in juli) | Smeets & van den Broeke (2008) als z0m > 1 mm, anders Andreas (1987) |
| AWS6 | seizoensgemiddelde van de SHR-mast (benadering, want S6 heeft geen mast) | Andreas (1987) |
| AWS9, AWS10 | constant 0.126 mm, zoals in het paper | Andreas (1987) |

Het paper bepaalt z0m bij S5 en S6 uit metingen op twee hoogtes. De PANGAEA-bestanden hebben maar één meethoogte, daarom komt z0m hier uit de eddy-covariancemasten.

De kolom met latente warmte is in de SHR-bestanden helemaal leeg. Daarom is LHF berekend uit de windschuifsnelheid, de stabiliteitscorrectie en het vochtverschil tussen lucht en oppervlak, die wel in de bestanden staan. Ter controle geeft dezelfde formule de gemeten voelbare warmte terug met r = 0.95.

### Wat er gevonden is

**Jaarlijkse smelt.** Die komt goed overeen met wat er over de K-transect bekend is:

| Station | Smelt per jaar (m w.e.) |
|---|---|
| AWS5 | 3.1 – 5.9 |
| AWS6 | 1.0 – 3.1 |
| AWS9 | 0.5 – 1.9 |
| AWS10 | 0.3 – 0.8 |

Dat past ook bij Kuipers Munneke et al. (2018): gemiddeld ~3.9 m w.e. per jaar bij S5 en ~2.0 bij S6 over 2003–2016.

**Berekend tegen gemeten.** De berekende smelt is vergeleken met de gemeten daling van het oppervlak (draadmeter of akoestische hoogtemeter, omgerekend met een ijsdichtheid van 900 kg/m³). Per maand:

| Station | r | Gemiddelde afwijking |
|---|---|---|
| AWS5 | 0.96 | +61 mm |
| AWS6 | 0.89 | −26 mm |
| AWS9 | 0.19 | −241 mm |
| AWS10 | 0.62 | −152 mm |

Hoofdstuk 3 geeft deze cijfers met betrouwbaarheidsintervallen.

Voor beide reeksen geldt dezelfde aanpak bij ontbrekende data: een periode telt alleen mee bij minstens 80% dekking. De waarde wordt dan opgeschaald van de tijd met geldige data naar de hele periode.

Bij AWS5 en AWS6 werkt de methode goed ([figuur](output/melt_calc_vs_obs_M.png)). Bij AWS9 en AWS10 meet de hoogtemeter ook sneeuwval en de hoogte van de sneeuw, en die verlaging is omgerekend alsof het ijs is. De slechte overeenkomst daar komt dus vooral door de meting, niet door de berekening.

**SHR tegen AWS5.** In juni tot 10 augustus 2021 komt AWS5 uit op 2640 mm w.e. en SHR op 854 mm w.e. ([figuur](output/zomer2021/melt_timeseries_D.png)). SHR smelt duidelijk minder, vooral omdat het oppervlak daar vaker net onder 0 °C blijft.

**Waar de smeltenergie vandaan komt.** Dat verschilt sterk met de hoogte. Bij AWS5 komt bijna de helft (48%) uit voelbare warmte, de warme lucht boven het ijs. Hoger op de ijskap is netto kortgolvige straling de belangrijkste bron, met 80% bij AWS6 en 87–90% bij AWS9 en AWS10.

**Voelbare warmte vergeleken met het paper.** Bij AWS6, AWS9 en AWS10 komt de zomergemiddelde voelbare warmte (22, 12 en 7 W/m²) goed overeen met het paper. Bij AWS5 is hij met ~95 W/m² hoger dan de ~55–90 W/m² in het paper. Dat verschil verdween niet met de nieuwe ruwheid, dus de oorzaak ligt waarschijnlijk ergens anders (bijvoorbeeld meethoogte of datacorrecties).

---

## 2. Hoogte en albedo

### Wat er gedaan is

Voor elk station en elke zomer (juni–augustus, JJA) zijn gemiddeld: temperatuur, wind, straling, albedo, de energiebalanstermen en de totale smelt. Alleen zomers met minstens 80% data tellen mee. Daarmee zijn drie dingen gedaan:

1. **Hoogtegradiënten:** lineaire regressie van elke grootheid tegen de hoogte. Dat is gedaan over alle jaren en over alleen 2012–2015, de jaren waarin alle vier stations data hebben.
2. **Albedo-experiment:** de smelt is opnieuw berekend met voor elke dag het gemiddelde albedo over alle jaren. Het verschil met de echte smelt is de directe bijdrage van een afwijkend albedo in dat jaar.
3. **Trends en controle op de stationswissel:** trends per station, ook over alleen de jaren vóór de wissel naar het nieuwe stationstype (2014–2016, verschilt per station).

### Wat er gevonden is

**Verandering met de hoogte** ([figuur temperatuur en albedo](output/elevation/elevation_T_albedo.png), [figuur energiebalans en smelt](output/elevation/elevation_gradients.png)):

| Grootheid | Verandering per 100 m hoger | R² |
|---|---|---|
| Luchttemperatuur | −0.51 °C | 0.99 |
| Albedo | +0.018 | 0.90 |
| Inkomende kortgolvige straling | +4.0 W/m² | 0.98 |
| Netto kortgolvige straling | −3.6 W/m² | 0.84 |
| Voelbare warmte | −6.2 W/m² | 0.78 |
| Zomersmelt | −228 mm w.e. | 0.95 |

Met maar vier stations zijn deze gradiënten onzeker. Hoofdstuk 3 geeft de betrouwbaarheidsintervallen.

- **De smeltgradiënt is stabiel:** in elk jaar ligt hij tussen −210 en −310 mm w.e. per 100 m.
- **Voelbare warmte past niet bij een rechte lijn:** AWS5 springt eruit met ~95 W/m², tegen ~22 W/m² bij AWS6.
- **Het albedo neemt niet gewoon toe met de hoogte:** in augustus is AWS6 het donkerst van alle stations (albedo ~0.45, tegen ~0.52 bij AWS5). Dat past bij de bekende donkere zone van de K-transect ([figuur](output/elevation/albedo_climatology.png)).

**Albedo-experiment** ([figuur](output/elevation/albedo_contribution.png)):

| Station | Deel van de verschillen tussen jaren dat door albedo komt | Grootste bijdrage |
|---|---|---|
| AWS5 (500 m) | 9% | +290 mm (2020) |
| AWS6 (1000 m) | 54% | +752 mm (2019) |
| AWS9 (1500 m) | 79% | +613 mm (2012) |
| AWS10 (1850 m) | 46% (5 jaar) | +161 mm (2012) |

- **Laag op de ijskap (AWS5):** daar ligt de hele zomer kaal ijs met een vrij constant albedo. Hoeveel het in een jaar smelt, hangt daar vooral af van temperatuur en voelbare warmte.
- **Rond de sneeuwgrens (AWS9):** daar telt vooral wanneer de sneeuw verdwijnt. Warme jaren zoals 2010 en 2012 worden er vooral door het albedo versterkt.

**Trends en meetartefact.** Op het eerste gezicht koelt AWS5 significant af (−0.74 °C per decennium) en neemt de voelbare warmte af. Dat blijkt een gevolg van de wissel van stationstype:
- **Wind:** de windsnelheid daalt bij alle vier stations met 0.6–1.0 m/s precies bij de wissel.
- **AWS5:** de temperatuur daalt 0.9 °C en de voelbare warmte 32 W/m² ten opzichte van AWS6. AWS6 zelf verandert in die jaren niet.

Over alleen de jaren vóór de wissel is er geen trend in temperatuur of voelbare warmte. Wel neemt de netto kortgolvige straling toe: +24 W/m² per decennium bij AWS5 (p = 0.009) en +40 W/m² bij AWS9 (p = 0.025), waar ook het albedo daalt. Dat wijst op een donkerder oppervlak. De reeks is wel kort (2004–2014/2016), dus dit is een voorzichtige conclusie.

---

## 3. Foutmarges en correlaties

### Wat er gedaan is

1. **Gevoeligheid van de zomersmelt.** De JJA-smelt is opnieuw berekend met één verstoring tegelijk. Voor de meetfouten zijn de nauwkeurigheden uit Kuipers Munneke et al. (2018, tabel 2) gebruikt:

   | Verstoring | Reden |
   |---|---|
   | luchttemperatuur ±0.4 °C | nauwkeurigheid temperatuursensor |
   | wind ±0.3 m/s | nauwkeurigheid windmeter |
   | kortgolvige straling ±5% | kalibratie stralingsmeter |
   | inkomende langgolvige straling ±5 W/m² | kalibratie (aanname) |
   | z0m ×2 en ÷2 | onzekerheid ruwheid |
   | meethoogte oud stationstype 2.5 en 4 m (standaard 3 m) | echte hoogte onbekend |
   | smeltdrempel 0.25 en 1 K (standaard 0.5 K) | modelkeuze |

   De totale foutmarge is de wortel van de kwadratensom van de afzonderlijke effecten, in de aanname dat de fouten onafhankelijk zijn.
2. **Validatie met betrouwbaarheidsintervallen.** Voor berekend tegen gemeten smelt per maand: r met 95%-interval (Fisher-z), en de gemiddelde afwijking met 95%-interval (t-verdeling).
3. **Correlaties zoals tabel 3 van het paper.** De correlatie tussen de JJA-smelt en elke energiebalansterm, met 95%-interval. Dit is gedaan over alle jaren en over alleen de jaren van het oude stationstype, dat het meest lijkt op de periode van het paper.
4. **Onzekerheid van de hoogte-analyse.** 95%-intervallen voor de hoogtegradiënten (t-verdeling, met maar 4 stations dus 2 vrijheidsgraden), en een bootstrap-interval voor het deel van de smeltvariantie dat door albedo komt.

### Wat er gevonden is

**Foutmarge van de zomersmelt** ([figuur](output/uncertainty/sensitivity.png)):

| Station | Gem. JJA-smelt | Foutmarge totaal | Zonder smeltdrempel |
|---|---|---|---|
| AWS5 | 3559 mm w.e. | ±23% | ±13% |
| AWS6 | 1697 mm w.e. | ±25% | ±8% |
| AWS9 | 951 mm w.e. | ±24% | ±8% |
| AWS10 | 396 mm w.e. | ±27% | ±11% |

- **De smeltdrempel is overal de grootste foutbron** (±19–24%). Of een tijdstap als smeltend telt, hangt af van de oppervlaktetemperatuur uit LWU binnen een paar tienden van een graad. Het paper heeft dit probleem niet, omdat het de oppervlaktetemperatuur met een model berekent.
- **Bij AWS5 komt de ruwheid op de tweede plaats:** +13% bij z0m ×2 en −6% bij z0m ÷2. Daarna volgt de temperatuur (±6%).
- **Hoger op de ijskap** wegen de straling en de temperatuur het zwaarst (elk ±4–6%). Ruwheid, wind en meethoogte doen daar nauwelijks iets (<1%).

**Validatie met betrouwbaarheidsintervallen** (per maand, maanden met meer dan 50 mm):

| Station | r [95%-interval] | Gemiddelde afwijking [95%-interval] | Ten opzichte van gemeten |
|---|---|---|---|
| AWS5 | 0.96 [0.94 – 0.97] | +61 mm [+25 – +97] | +11% |
| AWS6 | 0.89 [0.82 – 0.93] | −26 mm [−59 – +8] | −7% |
| AWS9 | 0.19 [−0.01 – 0.38] | −241 mm [−299 – −183] | −67% |
| AWS10 | 0.62 [0.22 – 0.84] | −152 mm [−226 – −78] | −70% |

- **AWS5:** de berekende smelt is significant hoger dan de gemeten smelt, maar de afwijking (+11%) valt ruim binnen de foutmarge van ±13–23%.
- **AWS6:** de afwijking is niet significant.
- **AWS9 en AWS10:** de grote afwijkingen komen door de meting (sneeuw omgerekend als ijs), zie punt 5 van de zwaktes.

**Correlaties met de JJA-smelt, vergeleken met het paper** (alleen jaren met het oude stationstype; r [95%-interval], met tussen haakjes de waarde uit het paper):

| | AWS5 (n = 12) | AWS6 (n = 9) | AWS9 (n = 10) |
|---|---|---|---|
| Netto kortgolvig | 0.46 [−0.12 – 0.81] (0.57) | 0.50 [−0.24 – 0.88] (0.78) | 0.89 [0.59 – 0.97] (0.88) |
| Netto langgolvig | 0.26 [−0.34 – 0.71] (0.58) | 0.67 [0.00 – 0.92] (0.66) | 0.36 [−0.35 – 0.81] (0.59) |
| Netto straling | 0.66 [0.17 – 0.89] (0.86) | 0.75 [0.17 – 0.94] (**0.98**) | 0.98 [0.90 – 0.99] (0.98) |
| Voelbare warmte | 0.69 [0.23 – 0.90] (0.78) | 0.73 [0.12 – 0.94] (0.74) | 0.80 [0.34 – 0.95] (0.71) |
| Latente warmte | 0.59 [0.06 – 0.86] (**0.88**) | 0.69 [0.04 – 0.93] (0.78) | 0.73 [0.20 – 0.93] (0.60) |
| Albedo | −0.43 [−0.79 – 0.16] (−0.76) | −0.67 [−0.92 – −0.02] (**−0.93**) | −0.96 [−0.99 – −0.85] (−0.96) |

AWS10 staat er niet in: met 4 jaar zijn de intervallen bijna even breed als het hele bereik van r.

- **Bijna alle waarden uit het paper vallen binnen onze 95%-intervallen.** Alleen drie (vetgedrukt) liggen er net buiten. Met 9–12 jaar zijn de verschillen met het paper dus grotendeels niet significant.
- **De stationswissel maakt veel uit.** Over alle jaren, inclusief het nieuwe stationstype, is de correlatie tussen voelbare warmte en smelt bij AWS6 −0.35 [−0.73 – 0.19] in plaats van 0.73.
- **AWS9 is het robuustst:** albedo en netto straling verklaren daar bijna alle verschillen tussen jaren, net als in het paper.

**Onzekerheid van de hoogtegradiënten** (4 stations):

| Grootheid | Gradiënt per 100 m [95%-interval] | p |
|---|---|---|
| Luchttemperatuur | −0.51 °C [−0.65 – −0.37] | 0.004 |
| Albedo | +0.018 [0.000 – 0.036] | 0.049 |
| Netto kortgolvige straling | −3.6 W/m² [−8.3 – +1.1] | 0.08 |
| Voelbare warmte | −6.2 W/m² [−16.0 – +3.7] | 0.11 |
| Zomersmelt | −228 mm w.e. [−394 – −61] | 0.03 |

Alleen de gradiënten van temperatuur, albedo en smelt zijn significant. Die van netto kortgolvige straling en voelbare warmte kunnen met vier stations niet van nul worden onderscheiden.

**Onzekerheid van het albedo-aandeel** (bootstrap over de jaren):

| Station | Deel van de smeltvariantie door albedo [95%-interval] |
|---|---|
| AWS5 | 9% [−5 – 25%] |
| AWS6 | 54% [10 – 76%] |
| AWS9 | 79% [71 – 85%] |
| AWS10 | 46% (5 jaar, interval niet zinvol) |

Bij AWS9 is het grote albedo-aandeel een robuuste uitkomst. Bij AWS6 is het interval breed, en bij AWS5 niet van nul te onderscheiden.

### Wat dit betekent

- **Robuust:**
  - de smelt neemt af met de hoogte (~−230 mm w.e. per 100 m);
  - bij AWS9 bepaalt het albedo de verschillen tussen jaren;
  - bij AWS5 is voelbare warmte een belangrijke smeltbron die meebeweegt met de smelt.
- **Onzeker:**
  - de gradiënten van straling en voelbare warmte;
  - het albedo-aandeel bij AWS6;
  - alles bij AWS10.
- **Verbetering:** de berekening wordt vooral zekerder als de smeltdrempel vervangen wordt door een berekende oppervlaktetemperatuur, zoals in het paper.

---

## Beperkingen en zwaktes in de code

De punten staan per groep, binnen elke groep van grootste naar kleinste invloed op de resultaten. Verwijzingen gaan naar functies in [code/melt_comparison.py](code/melt_comparison.py) en [code/elevation_albedo.py](code/elevation_albedo.py).

### Grote invloed op de uitkomst

1. **De ruwheid volgt het paper, maar niet helemaal** (`Z0M`, `SCALAR_SCHEME`, `z0m_series`, `scalar_roughness`).
   - z0m komt uit de eddy-covariancemasten, niet uit metingen op twee hoogtes zoals in het paper. Buiten 2019–2021 is het een seizoensgemiddelde, dus verschillen tussen jaren in ruwheid zitten er niet in.
   - Voor AWS6 is de ruwheid van SHR gebruikt, omdat S6 geen eigen mast heeft.
   - De coëfficiënten van Smeets & van den Broeke (2008) zijn overgenomen uit de literatuur en nog niet nagecontroleerd met het originele artikel.
   - Bij AWS5 blijft de voelbare warmte (~95 W/m²) hoger dan in het paper (~55–90 W/m²).
   - De smelt bij AWS5 verandert +13% / −6% bij een factor 2 in z0m (hoofdstuk 3).
2. **De meethoogte staat vast op 3 m voor het oude stationstype** (`Z_DEFAULT`). De werkelijke hoogtes staan in de metadata-xlsx van PANGAEA, maar worden niet gebruikt. Wind en temperatuur worden ook op verschillende hoogtes gemeten.
3. **Geen warmtestroom in het ijs, en een harde smeltdrempel.**
   - Alle energie bij een oppervlak op 0 °C gaat naar smelt. In het voorjaar moet eerst het koude ijs opwarmen, dus de smelt wordt dan overschat.
   - Of er smelt is, hangt af van de oppervlaktetemperatuur binnen 0.5 K van het smeltpunt (`T_MELT_TOL`). Die temperatuur komt uit LWU met emissiviteit 1. Een meetfout van enkele W/m² in LWU kan daardoor het verschil maken tussen wel en geen smelt. Bij SHR vallen veel dagen net onder de drempel. Dit is de grootste foutbron: een drempel van 0.25 of 1 K verandert de zomersmelt met ±19–24% (hoofdstuk 3).
4. **SHR heeft geen eigen stralingsmeting** (`melt_flux`). SWD, SWU en LWD komen van AWS5, en het albedo van AWS5 past waarschijnlijk niet bij SHR. Uitspraken over hoeveel SHR smelt zijn daardoor zwak. Data van KAN_L (PROMICE) zou dit oplossen.
5. **De gemeten smelt is alleen bruikbaar bij kaal ijs.** De akoestische hoogtemeter meet ook sneeuwval, maar alles wordt omgerekend met ijsdichtheid (`RHO_ICE`). Sneeuwval telt dan als negatieve smelt. Daarom is de validatie bij AWS9 en AWS10 niet bruikbaar.

### Ontbrekende data en aggregatie

6. **Het opvullen van gaten gaat ervan uit dat ze willekeurig vallen** (`aggregate`). Gaten krijgen de gemiddelde smeltsnelheid van de rest van de periode. Vallen ze vooral overdag, dan klopt de smelt van die periode niet.
7. **Alle tijdstappen tellen even zwaar.** In een periode met zowel 30- als 60-minutendata (rond een stationswissel) tellen de 30-minutenstappen dubbel. Aan de rand van een gat wordt de dekking iets overschat, omdat elke stap tot 1 uur wordt afgekapt.
8. **Sprongen in de hoogtemeting worden herkend met een vaste drempel van 5 cm per stap** (`observed_lowering`). Een herinstallatie met een sprong kleiner dan 5 cm telt dus als smelt of als aangroei.

### Natuurkundige vereenvoudigingen

9. **De iteratie voor de turbulente fluxen wordt niet op convergentie gecontroleerd** (`bulk_fluxes`). Er zijn altijd precies 10 stappen, en de stabiliteitsparameter wordt afgekapt tussen −5 en 10. Bij zeer stabiele lucht en weinig wind kan het resultaat onzin zijn zonder dat het opvalt.
10. **De keuze tussen verdampings- en sublimatiewarmte is inconsistent.** Een oppervlak op −0.3 °C telt als smeltend, maar krijgt sublimatiewarmte. Het verschil is ongeveer 13% van LHF.
11. **Geen correcties op de straling.** De scheefstand van de mast wordt niet gecorrigeerd, en albedowaarden boven 1 bij een lage zon worden niet weggefilterd.
12. **Het albedo-experiment meet alleen het directe stralingseffect.** De oppervlaktetemperatuur en de turbulente fluxen blijven gelijk, dus terugkoppelingen via die termen zitten er niet in.

### Analyse van hoogte en albedo

13. **De wissel van stationstype wordt gemeten, maar niet gecorrigeerd.** Trends over 2003–2021 bevatten daardoor sprongen. Ook de zomergemiddelden en gradiënten van verschillende periodes zijn niet helemaal vergelijkbaar.
14. **Het klimatologische albedo bevat ook het jaar zelf** (`climatological_albedo`). Bij AWS10, met maar 5 jaar, trekt een extreem jaar zijn eigen klimatologie mee, zodat het albedo-effect wordt onderschat. Het jaar zelf weglaten zou beter zijn.
15. **De JJA-berekening is een omweg** (`jja_summary`). Er wordt een jaartotaal gemaakt en dat wordt teruggeschaald naar JJA. Dat werkt, maar gaat stuk zodra iemand `aggregate()` aanpast.
16. **Trends worden al vanaf 5 jaar berekend** (`trends`). Bij AWS10 komt er zo een "trend" van −5 °C per decennium in de uitvoer, zonder waarschuwing.

### Robuustheid van de code

17. **`--common` laat stations zonder data stilletjes weg.** Bijvoorbeeld AWS10 in 2021: in de uitvoer staat gewoon NaN, zonder melding.
18. **Instellingen staan vast in de code.** Ruwheid, drempels en meethoogte zijn globale constanten, zonder optie op de opdrachtregel. De gevoeligheidsanalyse in `uncertainty.py` past ze daarom tijdelijk aan in het geheugen.
19. **Geen tests, en dezelfde data wordt meerdere keren ingelezen.** Er zijn geen controles op bijvoorbeeld energiebehoud, eenheden of een bekende dag. `melt_flux` leest bij elke aanroep alle AWS5-data opnieuw in.
20. **Niet gebruikte data:** de bestanden `grl_aws*_final_year*.txt` hebben geen kolomnamen. Daardoor ontbreekt 2022.

## Mogelijke vervolgstappen

1. **Oppervlaktetemperatuur berekenen in plaats van een smeltdrempel:** de oppervlaktetemperatuur per tijdstap oplossen uit de energiebalans, met een eenvoudig model voor warmtegeleiding in het ijs, zoals in het paper. Dat haalt de grootste foutbron weg (punt 3 hierboven).
2. **De voelbare warmte bij AWS5 verklaren:** de berekende voelbare warmte vergelijken met de gemeten voelbare warmte van de S5-mast (2019–2021). Daarnaast de echte meethoogtes uit de PANGAEA-metadata gebruiken (punt 2).
3. **AWS5 vergelijken met SHR:** met stralingsdata van KAN_L kan SHR echt met AWS5 worden vergeleken.
4. **Betere validatie hoger op de ijskap:** onderscheid maken tussen sneeuw en ijs bij de omrekening van de gemeten daling, zodat AWS9 en AWS10 ook te valideren zijn.

## Reproduceren

```bash
python code/melt_comparison.py --freq M          # smelt per maand (ook: h, D, W, Y)
python code/melt_comparison.py --freq D --start 2021-06-01 --end 2021-08-10 --out output/zomer2021
python code/elevation_albedo.py                  # hoogte- en albedo-analyse
python code/uncertainty.py                       # foutmarges en correlaties (na de twee scripts hierboven)
```
