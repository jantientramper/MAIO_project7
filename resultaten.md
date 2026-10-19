# Resultaten: smelt langs de K-transect

Dit document vat samen wat er met de stationsdata in `data/` is gedaan en wat daaruit kwam. Het gaat om twee delen:

1. **Smelt per station**: een vergelijking van de berekende smelt tussen alle stations ([code/melt_comparison.py](code/melt_comparison.py)).
2. **Hoogte en albedo**: hoe de energiebalans en de smelt veranderen met de hoogte, en welke rol het albedo speelt ([code/elevation_albedo.py](code/elevation_albedo.py)).

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
| S5 (= locatie AWS5) | 500 m | van AWS5 | SHF gemeten (eddy covariance), LHF zelf berekend |
| SHR (bij KAN_L) | niet in data | van AWS5 (benadering) | SHF gemeten, LHF zelf berekend |

De kolom met latente warmte is in de S5- en SHR-bestanden helemaal leeg. Daarom is LHF berekend uit de windschuifsnelheid, de stabiliteitscorrectie en het vochtverschil tussen lucht en oppervlak, die wel in de bestanden staan. Ter controle geeft dezelfde formule de gemeten voelbare warmte terug met r = 0.95.

### Wat er gevonden is

**Jaarlijkse smelt.** Die komt goed overeen met wat er over de K-transect bekend is:

| Station | Smelt per jaar (m w.e.) |
|---|---|
| AWS5 | 3.0 – 5.9 |
| AWS6 | 1.1 – 3.2 |
| AWS9 | 0.5 – 2.0 |
| AWS10 | 0.3 – 0.9 |

**Berekend tegen gemeten.** De berekende smelt is vergeleken met de gemeten daling van het oppervlak (draadmeter of akoestische hoogtemeter, omgerekend met een ijsdichtheid van 900 kg/m³). Per maand:

| Station | r | Gemiddelde afwijking |
|---|---|---|
| AWS5 | 0.96 | +64 mm |
| AWS6 | 0.89 | +9 mm |
| AWS9 | 0.22 | −221 mm |
| AWS10 | 0.60 | −136 mm |

Bij AWS5 en AWS6 werkt de methode goed ([figuur](output/melt_calc_vs_obs_M.png)). Bij AWS9 en AWS10 meet de hoogtemeter ook sneeuwval en de hoogte van de sneeuw, en die verlaging is omgerekend alsof het ijs is. De slechte overeenkomst daar komt dus vooral door de meting, niet door de berekening.

**S5 tegen AWS5.** Dit is dezelfde plek, maar met onafhankelijk bepaalde turbulente fluxen. In juni tot 10 augustus 2021 komt S5 uit op 2413 mm w.e. en AWS5 op 2574 mm w.e. ([figuur](output/zomer2021/melt_timeseries_D.png)). SHR smelt in dezelfde periode duidelijk minder (854 mm w.e.), vooral omdat het oppervlak daar vaker net onder 0 °C blijft.

**Waar de smeltenergie vandaan komt.** Dat verschilt sterk met de hoogte. Bij AWS5 komt bijna de helft (46%) uit voelbare warmte, de warme lucht boven het ijs. Hoger op de ijskap is netto kortgolvige straling de belangrijkste bron, met 79% bij AWS6 en 88–90% bij AWS9 en AWS10.

**Grootste onzekerheid.** Dat is de ruwheid van het oppervlak (z0m), die vast is gekozen. Bij AWS5 in 2019 loopt de jaarsmelt van 3.7 naar 5.7 m w.e. als z0m van 0.001 naar 0.03 m gaat.

---

## 2. Hoogte en albedo

### Wat er gedaan is

Voor elk station en elke zomer (juni–augustus, JJA) zijn gemiddeld: temperatuur, wind, straling, albedo, de energiebalanstermen en de totale smelt. Alleen zomers met minstens 80% data tellen mee. Daarmee zijn drie dingen gedaan:

1. **Hoogtegradiënten:** lineaire regressie van elke grootheid tegen de hoogte. Dat is gedaan over alle jaren en over alleen 2012–2015, de jaren waarin alle vier stations data hebben.
2. **Albedo-experiment:** de smelt is opnieuw berekend met voor elke dag het gemiddelde albedo over alle jaren. Het verschil met de echte smelt is de directe bijdrage van een afwijkend albedo in dat jaar.
3. **Trends en controle op de stationswissel:** trends per station, ook over alleen de jaren vóór de wissel naar het nieuwe stationstype (2014–2016, verschilt per station).

### Wat er gevonden is

**Verandering met de hoogte** ([figuur](output/elevation/elevation_gradients.png)):

| Grootheid | Verandering per 100 m hoger | R² |
|---|---|---|
| Luchttemperatuur | −0.51 °C | 0.99 |
| Albedo | +0.018 | 0.90 |
| Inkomende kortgolvige straling | +4.0 W/m² | 0.98 |
| Netto kortgolvige straling | −3.6 W/m² | 0.84 |
| Voelbare warmte | −5.6 W/m² | 0.85 |
| Zomersmelt | −222 mm w.e. | 0.97 |

- **De smeltgradiënt is stabiel:** in elk jaar ligt hij tussen −200 en −300 mm w.e. per 100 m.
- **Voelbare warmte past niet bij een rechte lijn:** AWS5 springt eruit met ~90 W/m², tegen ~30 W/m² bij AWS6.
- **Het albedo neemt niet gewoon toe met de hoogte:** in augustus is AWS6 het donkerst van alle stations (albedo ~0.45, tegen ~0.52 bij AWS5). Dat past bij de bekende donkere zone van de K-transect ([figuur](output/elevation/albedo_climatology.png)).

**Albedo-experiment** ([figuur](output/elevation/albedo_contribution.png)):

| Station | Deel van de verschillen tussen jaren dat door albedo komt | Grootste bijdrage |
|---|---|---|
| AWS5 (500 m) | 9% | +290 mm (2020) |
| AWS6 (1000 m) | 52% | +752 mm (2019) |
| AWS9 (1500 m) | 77% | +613 mm (2012) |
| AWS10 (1850 m) | 43% (5 jaar) | +161 mm (2012) |

- **Laag op de ijskap (AWS5):** daar ligt de hele zomer kaal ijs met een vrij constant albedo. Hoeveel het in een jaar smelt, hangt daar vooral af van temperatuur en voelbare warmte.
- **Rond de sneeuwgrens (AWS9):** daar telt vooral wanneer de sneeuw verdwijnt. Warme jaren zoals 2010 en 2012 worden er vooral door het albedo versterkt.

**Trends en meetartefact.** Op het eerste gezicht koelt AWS5 significant af (−0.74 °C per decennium) en neemt de voelbare warmte af. Dat blijkt een gevolg van de wissel van stationstype:
- **Wind:** de windsnelheid daalt bij alle vier stations met 0.6–1.0 m/s precies bij de wissel.
- **AWS5:** de temperatuur daalt 0.9 °C en de voelbare warmte 30 W/m² ten opzichte van AWS6. AWS6 zelf verandert in die jaren niet.

Over alleen de jaren vóór de wissel is er geen trend in temperatuur of voelbare warmte. Wel neemt de netto kortgolvige straling toe: +24 W/m² per decennium bij AWS5 (p = 0.009) en +40 W/m² bij AWS9 (p = 0.025), waar ook het albedo daalt. Dat wijst op een donkerder oppervlak. De reeks is wel kort (2004–2014/2016), dus dit is een voorzichtige conclusie.

---

## Beperkingen

- **SHR heeft geen eigen stralingsmeting.** De straling en het albedo komen van AWS5, wat voor SHR een benadering is. Data van KAN_L (PROMICE) zou dit oplossen.
- **Vaste parameters.** De ruwheidslengte per station, de meethoogte (3 m voor het oude stationstype) en de smeltdrempel (0.5 K) zijn vast gekozen. De ruwheid heeft de meeste invloed op de uitkomst.
- **Geen warmtestroom in het ijs.** Daardoor wordt de smelt in het voorjaar waarschijnlijk iets overschat, omdat dan eerst het ijs moet opwarmen.
- **Het albedo-experiment meet alleen het directe effect.** De oppervlaktetemperatuur en de turbulente fluxen blijven gelijk, dus terugkoppelingen via die termen zitten er niet in.
- **Trends over 2003–2021 zijn niet zonder meer bruikbaar.** De wissel van stationstype veroorzaakt sprongen in de reeks.
- **Niet gebruikt:** de bestanden `grl_aws*_final_year*.txt` hebben geen kolomnamen. Daardoor ontbreekt 2022.

## Mogelijke vervolgstappen

1. **Ruwheid bepalen bij S5:** de S5-fluxbestanden bevatten zelf waarden voor z0m. Daarmee kan de grootste onzekerheid in de smelt bij AWS5 kleiner worden (onderzoekslijn A).
2. **S5 vergelijken met SHR:** met stralingsdata van KAN_L kan SHR echt met S5 worden vergeleken.
3. **Betere validatie hoger op de ijskap:** onderscheid maken tussen sneeuw en ijs bij de omrekening van de gemeten daling, zodat AWS9 en AWS10 ook te valideren zijn.

## Reproduceren

```bash
python code/melt_comparison.py --freq M          # smelt per maand (ook: h, D, W, Y)
python code/melt_comparison.py --freq D --start 2021-06-01 --end 2021-08-10 --out output/zomer2021
python code/elevation_albedo.py                  # hoogte- en albedo-analyse
```
