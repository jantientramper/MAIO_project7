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
| SHR (bij KAN_L) | niet in data | van AWS5 (benadering) | SHF gemeten (eddy covariance), LHF zelf berekend |

De eddy-covariancemast S5 staat op dezelfde plek als AWS5 en is daarom niet als apart station meegenomen.

De kolom met latente warmte is in de SHR-bestanden helemaal leeg. Daarom is LHF berekend uit de windschuifsnelheid, de stabiliteitscorrectie en het vochtverschil tussen lucht en oppervlak, die wel in de bestanden staan. Ter controle geeft dezelfde formule de gemeten voelbare warmte terug met r = 0.95.

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
| AWS5 | 0.96 | +53 mm |
| AWS6 | 0.89 | +5 mm |
| AWS9 | 0.19 | −233 mm |
| AWS10 | 0.62 | −148 mm |

Voor beide reeksen geldt dezelfde aanpak bij ontbrekende data: een periode telt alleen mee bij minstens 80% dekking. De waarde wordt dan opgeschaald van de tijd met geldige data naar de hele periode.

Bij AWS5 en AWS6 werkt de methode goed ([figuur](output/melt_calc_vs_obs_M.png)). Bij AWS9 en AWS10 meet de hoogtemeter ook sneeuwval en de hoogte van de sneeuw, en die verlaging is omgerekend alsof het ijs is. De slechte overeenkomst daar komt dus vooral door de meting, niet door de berekening.

**SHR tegen AWS5.** In juni tot 10 augustus 2021 komt AWS5 uit op 2574 mm w.e. en SHR op 854 mm w.e. ([figuur](output/zomer2021/melt_timeseries_D.png)). SHR smelt duidelijk minder, vooral omdat het oppervlak daar vaker net onder 0 °C blijft.

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

## Beperkingen en zwaktes in de code

De punten staan per groep, binnen elke groep van grootste naar kleinste invloed op de resultaten. Verwijzingen gaan naar functies in [code/melt_comparison.py](code/melt_comparison.py) en [code/elevation_albedo.py](code/elevation_albedo.py).

### Grote invloed op de uitkomst

1. **De ruwheidslengtes zijn geschat, en voor AWS5 liggen ze ver naast de metingen** (`Z0M`, `Z0H_RATIO`). De eddy-covariancemast op dezelfde locatie (`data/S5`) geeft voor de zomer:

   | | Gemeten (JJA-mediaan) | In het script |
   |---|---|---|
   | z0m | 0.027 m | 0.01 m |
   | z0h | 0.00005 m | 0.001 m (20× te groot) |

   De twee afwijkingen werken tegen elkaar in, maar die in z0h is waarschijnlijk de grootste. Dat kan de te hoge voelbare warmte en de positieve afwijking van +53 mm per maand bij AWS5 verklaren. De vaste verhouding `z0h = z0m/10` klopt voor ruw ijs niet: daar is z0h 2–3 ordes kleiner dan z0m.
2. **De meethoogte staat vast op 3 m voor het oude stationstype** (`Z_DEFAULT`). De werkelijke hoogtes staan in de metadata-xlsx van PANGAEA, maar worden niet gebruikt. Wind en temperatuur worden ook op verschillende hoogtes gemeten.
3. **Geen warmtestroom in het ijs, en een harde smeltdrempel.**
   - Alle energie bij een oppervlak op 0 °C gaat naar smelt. In het voorjaar moet eerst het koude ijs opwarmen, dus de smelt wordt dan overschat.
   - Of er smelt is, hangt af van de oppervlaktetemperatuur binnen 0.5 K van het smeltpunt (`T_MELT_TOL`). Die temperatuur komt uit LWU met emissiviteit 1. Een meetfout van enkele W/m² in LWU kan daardoor het verschil maken tussen wel en geen smelt. Bij SHR vallen veel dagen net onder de drempel.
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
18. **Instellingen staan vast in de code.** Ruwheid, drempels en meethoogte zijn globale constanten, zonder optie op de opdrachtregel. Een gevoeligheidsanalyse vraagt daardoor aanpassingen in de code.
19. **Geen tests, en dezelfde data wordt meerdere keren ingelezen.** Er zijn geen controles op bijvoorbeeld energiebehoud, eenheden of een bekende dag. `melt_flux` leest bij elke aanroep alle AWS5-data opnieuw in.
20. **Niet gebruikte data:** de bestanden `grl_aws*_final_year*.txt` hebben geen kolomnamen. Daardoor ontbreekt 2022.

## Mogelijke vervolgstappen

1. **Ruwheid bepalen bij AWS5:** de eddy-covariancebestanden van dezelfde locatie (`data/S5`) bevatten gemeten z0m en z0h. Met die waarden kan de grootste fout in de smelt bij AWS5 kleiner worden (onderzoekslijn A, zie punt 1 hierboven). Dat gaat het best samen met instellingen vanaf de opdrachtregel (punt 18), zodat er meteen een gevoeligheidsanalyse kan draaien.
2. **AWS5 vergelijken met SHR:** met stralingsdata van KAN_L kan SHR echt met AWS5 worden vergeleken.
3. **Betere validatie hoger op de ijskap:** onderscheid maken tussen sneeuw en ijs bij de omrekening van de gemeten daling, zodat AWS9 en AWS10 ook te valideren zijn.

## Reproduceren

```bash
python code/melt_comparison.py --freq M          # smelt per maand (ook: h, D, W, Y)
python code/melt_comparison.py --freq D --start 2021-06-01 --end 2021-08-10 --out output/zomer2021
python code/elevation_albedo.py                  # hoogte- en albedo-analyse
```
