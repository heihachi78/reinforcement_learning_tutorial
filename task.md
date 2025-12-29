Reinforcement learning
-	A megerősítéses tanulás jellemzői, amelyek megkülönböztetik más gépi tanulási paradigmáktól:
o	Nincs tanár (supervisor), csak jutalomsignál (reward)
o	A visszajelzés késleltetett, nem azonnali
o	Az idő számít – az adatok szekvenciálisak, nem függetlenek egymástól (nem i.i.d.)
o	Az ágens cselekedetei befolyásolják a későbbi adatokat, amelyeket kap
-	A jutalom (reward) Rₜ egy skalár visszajelző érték, amely megmutatja, mennyire jól teljesít az ágens a t időpillanatban. Az ágens célja a kumulatív jutalom maximalizálása.
-	A megerősítéses tanulás a jutalom-hipotézisen alapul:
o	„Minden cél leírható a várható kumulatív jutalom maximalizálásaként."
o	Ez vitatott állítás. Érvek mellette: sok komplex viselkedés valóban megfogalmazható jutalom-optimalizálásként. Érvek ellene: bizonyos célok (pl. kíváncsiság, kreativitás, etikai megfontolások) nehezen számszerűsíthetők egyetlen skalár jutalommal, és a jutalom megtervezése (reward shaping) gyakran nagyon nehéz feladat.
-	A szekvenciális döntéshozatal célja olyan cselekvések kiválasztása, amelyek maximalizálják a teljes jövőbeli jutalmat. A cselekvéseknek hosszú távú következményei lehetnek, és a jutalom késleltetve érkezhet – ezért néha érdemes feláldozni az azonnali jutalmat a nagyobb hosszú távú nyereségért.
o	Példák: pénzügyi befektetés (hónapokig érlelődik), helikopter tankolása (órákkal későbbi balesetet előzhet meg), ellenfél blokkolása sakkban (sok lépéssel későbbi győzelmet segíthet).
Az ügynök és a környezet
 
-	Az ágens és környezet interakciója minden t lépésben:
o	Az ágens
	Végrehajt egy Aₜ cselekvést
	Megkapja az Oₜ megfigyelést
	Megkapja az Rₜ skalár jutalmat
o	A környezet
	Fogadja az Aₜ cselekvést
	Kibocsátja az Oₜ₊₁ megfigyelést
	Kibocsátja az Rₜ₊₁ jutalmat
o	A t időlépés a környezet lépésénél növekszik. Ez egy folyamatos kölcsönhatás: az ágens cselekszik, a környezet reagál, és ez ismétlődik.
-	A történet (history) a megfigyelések, cselekvések és jutalmak sorozata:
o	Hₜ = O₁, R₁, A₁, ..., Aₜ₋₁, Oₜ, Rₜ
o	vagyis minden megfigyelhető változó t időpontig (például egy robot szenzomotoros adatfolyama)
-	A következő esemény a történettől függ: az ágens ez alapján választ cselekvést, a környezet pedig megfigyelést és jutalmat.
Állapot
-	Az állapot (state) az az információ, amely meghatározza, mi történik ezután. Formálisan az állapot a történet egy függvénye:
o	Sₜ = f(Hₜ)
o	Az állapot tehát a történet tömörített, releváns összefoglalása.
-	A környezet állapota (Sₜᵉ) a környezet belső, privát reprezentációja – azaz az az adat, amelyet a környezet használ a következő megfigyelés és jutalom meghatározásához. A környezet állapota általában nem látható az ágens számára. Még ha látható is lenne, gyakran irreleváns információkat is tartalmazhat.
-	Az ágens állapota (Sₜᵃ) az ágens belső reprezentációja – azaz az az információ, amelyet az ágens a következő cselekvés kiválasztásához használ. Ez az információ, amit a megerősítéses tanulási algoritmusok használnak. Az ágens állapota a történet bármilyen függvénye lehet:
o	Sₜᵃ = f(Hₜ)
Markov állapot
-	Az információs állapot (más néven Markov-állapot) a történet összes hasznos információját tartalmazza.
-	Definíció: Egy Sₜ állapot akkor és csak akkor Markov, ha:
o	P[Sₜ₊₁ | Sₜ] = P[Sₜ₊₁ | S₁, ..., Sₜ]
o	Ez azt jelenti: „A jövő független a múlttól, ha ismerjük a jelent."
-	Ha ismerjük az állapotot, a történet eldobható – az állapot elégséges statisztika a jövőre nézve.
o	A környezet állapota (Sₜᵉ) Markov, és a teljes történet (Hₜ) is Markov (triviálisan, hiszen mindent tartalmaz).
-	Teljesen megfigyelhető környezetek esetén az ágens közvetlenül látja a környezet állapotát:
o	Oₜ = Sₜᵃ = Sₜᵉ
o	Vagyis az ágens állapota = környezet állapota = információs állapot.
o	Formálisan ezt Markov döntési folyamatnak (MDP) nevezzük.
-	Részlegesen megfigyelhető környezetek esetén az ágens csak közvetetten látja a környezetet.
o	Példák: egy robot kamerával nem tudja a pontos pozícióját, egy kereskedő ágens csak az aktuális árakat látja, egy pókerjátékos csak a nyilvános lapokat
o	Ilyenkor az ágens állapota ≠ környezet állapota. Formálisan ezt részlegesen megfigyelhető Markov döntési folyamatnak (POMDP) nevezzük.
o	Az ágensnek saját állapotreprezentációt kell építenie, például:
	Teljes történet: Sₜᵃ = Hₜ
	Hiedelmek a környezet állapotáról: valószínűségi eloszlás a lehetséges állapotok felett
	Rekurrens neurális háló: Sₜᵃ = σ(Sₜ₋₁Wₛ + OₜWₒ)
RL ágens komponensei
-	Egy ágens tartalmazhatja ezek közül egyet, többet, vagy akár mindhármat (policy, value function, model).
Policy
-	Politika (Policy): az ágens viselkedési függvénye – meghatározza, milyen cselekvést válasszon
o	Egy leképezés az állapotból cselekvésre.
o	Determinisztikus politika
	a = π(s)
	egy adott állapothoz mindig ugyanaz a cselekvés tartozik
o	Sztochasztikus politika
	π(a|s) = P[Aₜ=a|Sₜ=s]
	az adott állapotban a cselekvések valószínűségi eloszlását adja meg
 
Value function
-	Értékfüggvény (Value function): megmutatja, mennyire jó egy adott állapot és/vagy cselekvés
o	A jövőbeli jutalom előrejelzése. Segítségével értékelhetjük az állapotok jóságát/rosszaságát, és ez alapján választhatunk cselekvések között.
o	Vπ(s)=E[Rₜ₊₁+γRₜ₊₂+γ²Rₜ₊₃+...|Sₜ=s]
	Ez a várható kumulatív diszkontált jutalom az s állapotból indulva, ahol γ (gamma) a diszkontálási tényező, amely a jövőbeli jutalmak súlyozását határozza meg (a távolabbi jutalmak kevésbé számítanak).
 
Model
-	Modell (Model): az ágens reprezentációja a környezetről – hogyan működik a világ
o	Azt jósolja meg, mit fog a környezet ezután tenni:
	P a következő állapotot jósolja
	R a következő (azonnali) jutalmat jósolja
 
	Tehát a modell tartalmazza az állapotátmenet-valószínűségeket és a várható jutalmakat – ez alapján az ágens „szimulálhatja", mi történne különböző cselekvések esetén.
 
Az RL ágensek kategorizálása
Az értékfüggvény és politika alapján
-	Érték-alapú (Value Based): csak értékfüggvény van, politika nincs (implicit)
-	Politika-alapú (Policy Based): csak politika van, értékfüggvény nincs
-	Actor-Critic: politika és értékfüggvény is van
A modell alapján
-	Modell-mentes (Model Free): politika és/vagy értékfüggvény, de nincs modell a környezetről
-	Modell-alapú (Model Based): politika és/vagy értékfüggvény, és van modell is
 
Szekvenciális döntéshozatal
-	A szekvenciális döntéshozatal célja olyan cselekvések kiválasztása, amelyek maximalizálják a teljes jövőbeli jutalmat. A cselekvéseknek hosszú távú következményei lehetnek, és a jutalom késleltetve érkezhet – ezért néha érdemes feláldozni az azonnali jutalmat a nagyobb hosszú távú nyereségért.
A szekvenciális döntéshozatal két alapvető problémája
Tanulás (Reinforcement Learning)
-	A környezet kezdetben ismeretlen
-	Az ágens interakcióba lép a környezettel
-	Az ágens javítja a politikáját tapasztalat alapján
Tervezés (Planning)
-	A környezet modellje ismert
-	Az ágens számításokat végez a modellel (külső interakció nélkül)
-	Az ágens javítja a politikáját gondolkodás útján
-	Más néven: deliberáció, következtetés, introspekció, keresés
Exploration vs Exploitation
-	A megerősítéses tanulás olyan, mint a próba-hiba alapú tanulás. Az ágensnek fel kell fedeznie egy jó politikát a környezettel szerzett tapasztalataiból, miközben nem veszíthet túl sok jutalmat a tanulás során. Ez a felfedezés és kiaknázás közötti egyensúly dilemmája.
-	Felfedezés (Exploration): több információt szerez a környezetről – új cselekvéseket próbál ki.
-	Kiaknázás (Exploitation): a már ismert információt használja a jutalom maximalizálására – a már bevált cselekvéseket választja.
-	Általában fontos mindkettőt alkalmazni: ha csak kiaknázunk, lemaradhatunk jobb lehetőségekről; ha csak felfedezünk, nem használjuk ki a már megszerzett tudást. A kettő közötti egyensúly megtalálása a megerősítéses tanulás egyik kulcskérdése.
Uniform random policy
-	A uniform random policy (egyenletes véletlenszerű politika) azt jelenti, hogy az ágens minden állapotban egyenlő valószínűséggel választ az összes lehetséges cselekvés közül.
o	Például ha 4 lehetséges cselekvés van (fel, le, balra, jobbra), akkor mindegyiket 25%-25%-25%-25% valószínűséggel választja, függetlenül attól, milyen állapotban van.
-	Ez a legegyszerűbb sztochasztikus politika, amely:
o	Nem használ semmilyen tanult tudást
o	Tisztán véletlenszerűen cselekszik
o	Gyakran kiindulópontként vagy összehasonlítási alapként (baseline) szolgál
o	Maximális felfedezést (exploration) biztosít, de nincs benne kiaknázás (exploitation)
Prediction vs Control
-	Predikció (Prediction): a jövő értékelése – adott politika mellett megbecsüljük a várható jutalmat.
-	Kontroll (Control): a jövő optimalizálása – megkeressük a legjobb politikát.
-	A kontroll gyakran predikcióra épül – ha meg tudjuk mondani, mennyire jó egy politika (predikció), akkor össze tudjuk hasonlítani a politikákat, és kiválaszthatjuk a legjobbat (kontroll).
Markov Decision Process
-	A Markov döntési folyamatok (MDP-k) formálisan írják le a környezetet a megerősítéses tanulásban, ahol a környezet teljesen megfigyelhető – vagyis az aktuális állapot teljesen jellemzi a folyamatot.
-	Szinte minden RL probléma megfogalmazható MDP-ként:
o	Az optimális irányítás főként folytonos MDP-kkel foglalkozik
o	A részlegesen megfigyelhető problémák átalakíthatók MDP-vé
o	A „bandit" problémák egyállapotú MDP-k
Markov állapot
-	Markov állapot: Egy Sₜ állapot akkor és csak akkor Markov, ha:
o	P[Sₜ₊₁ | Sₜ] = P[Sₜ₊₁ | S₁, ..., Sₜ]
o	Ez azt jelenti: „A jövő független a múlttól, ha ismerjük a jelent."
o	Az állapot tartalmazza az összes releváns információt a történetből.
o	Ha ismerjük az aktuális állapotot, a múltbeli történet eldobható – az állapot önmagában elegendő a jövő előrejelzéséhez.
Markov folyamat
-	Definíció: A Markov-folyamat (vagy Markov-lánc) egy (S, P) pár:
o	S: állapotok (véges) halmaza
o	P: állapotátmenet-valószínűségi mátrix, ahol Pₛₛ'=P[Sₜ₊₁=s'|Sₜ=s]
o	Tehát a Markov-folyamat leírja, hogyan lépked a rendszer véletlenszerűen állapotról állapotra, ahol a következő állapot csak az aktuális állapottól függ.
-	Az állapotátmenet-mátrix a Markov-folyamatokban:
o	Egy s Markov-állapotból egy s' utódállapotba való átmenet valószínűsége:
	Pₛₛ'=P[Sₜ₊₁=s'|Sₜ=s]
-	A P állapotátmenet-mátrix tartalmazza az összes s állapotból az összes s' állapotba való átmenet valószínűségét. A mátrix minden sora 1-re összegződik (mivel valahova biztosan átmegyünk). A sorok a kiinduló állapotokat, az oszlopok a célállapotokat jelölik.
 
 
-	A Markov-folyamat egy memória nélküli véletlen folyamat, vagyis véletlen állapotok sorozata (S₁, S₂, ...) Markov-tulajdonsággal.
Markov jutalom folyamat (MRP)
-	A Markov jutalom-folyamat (MRP) egy Markov-lánc értékekkel kiegészítve.
-	Definíció: A Markov jutalom-folyamat egy (S, P, R, γ) négyes:
o	S: állapotok véges halmaza
o	P: állapotátmenet-valószínűségi mátrix, Pₛₛ' = P[Sₜ₊₁ = s' | Sₜ = s]
o	R: jutalomfüggvény, Rₛ = E[Rₜ₊₁ | Sₜ = s] – a várható jutalom az adott állapotban
o	γ: diszkontálási tényező, γ ∈ [0, 1] – a jövőbeli jutalmak súlyozása
o	A Markov-folyamathoz képest itt már van jutalom is, de még nincs cselekvés (azt az MDP adja hozzá).
 
Return
-	A Gₜ hozam a t időponttól számított összes diszkontált jutalom:
 
-	A γ (gamma) diszkontálási tényező (γ ∈ [0, 1]) a jövőbeli jutalmak jelenértékét adja meg:
o	Egy k+1 lépés múlva kapott R jutalom értéke: γᵏR
o	Az azonnali jutalom értékesebb, mint a késleltetett
o	A γ értékének hatása:
	γ ≈ 0: „rövidlátó" értékelés – csak a közeli jutalmak számítanak
	γ ≈ 1: „távlátó" értékelés – a távoli jutalmak is fontosak
-	A legtöbb Markov jutalom- és döntési folyamatban diszkontálunk. Okai:
o	Matematikailag kényelmes – könnyebben kezelhetők a képletek
o	Elkerüli a végtelen hozamot ciklikus folyamatokban
o	A jövő bizonytalansága nem mindig reprezentálható teljesen
o	Pénzügyi szempontból az azonnali jutalom kamatozhat, így többet ér
o	Állatok és emberek viselkedése is az azonnali jutalmat részesíti előnyben
-	Néha lehet diszkontálás nélküli folyamatot is használni (γ = 1), például ha minden sorozat véges és terminál.
Value function
-	Az értékfüggvény (value function) v(s) megadja az s állapot hosszú távú értékét.
-	Definíció: Az MRP állapot-értékfüggvénye v(s) a várható hozam az s állapotból indulva:
o	v(s)=E[Gₜ|Sₜ=s]
o	Vagyis az értékfüggvény megmondja, hogy egy adott állapotból indulva összességében mennyi jutalmat várhatunk a jövőben.
Bellman-egyenlet MRP-kre
-	Az értékfüggvény két részre bontható:
o	Azonnali jutalom: Rₜ₊₁
o	Utódállapot diszkontált értéke: γv(Sₜ₊₁)
o	Tehát egy állapot értéke = az azonnali jutalom + a következő állapot diszkontált értéke (várható értékben).
-	A levezetés:
 
-	Ez egy rekurzív összefüggés: az állapot értékét a rákövetkező állapotok értékével fejezzük ki. Ez a Bellman-egyenlet alapgondolata, és kulcsfontosságú a megerősítéses tanulásban.
 
-	A Bellman-egyenlet mátrixos formában:
o	v=R+γPv
o	ahol:
	v: oszlopvektor, minden állapotra egy értékkel (az állapotértékek)
	R: oszlopvektor, minden állapotra a jutalom
	P: állapotátmenet-valószínűségi mátrix
	γ: diszkontálási tényező
 
-	A Bellman-egyenlet lineáris egyenlet, így közvetlenül megoldható:
o	v = R + γPv 
	(I - γP)v = R 
•	v=(I-γP)⁻¹R
o	I az egységmátrix (identity)
o	A közvetlen megoldás számítási komplexitása O(n³) n állapot esetén, ezért ez csak kis MRP-kre alkalmazható.
o	Nagy MRP-kre iteratív módszereket használnak:
	Dinamikus programozás
	Monte-Carlo értékelés
	Temporal-Difference (TD) tanulás
Markov döntési folyamat (MDP)
-	A Markov döntési folyamat (MDP) egy Markov jutalom-folyamat döntésekkel kiegészítve – olyan környezet, ahol minden állapot Markov.
-	Definíció: Az MDP egy (S, A, P, R, γ) ötös:
o	S: állapotok véges halmaza
o	A: cselekvések véges halmaza
o	P: állapotátmenet-valószínűségi mátrix, Pₛₛ'ᵃ = P[Sₜ₊₁ = s' | Sₜ = s, Aₜ = a]
o	R: jutalomfüggvény, Rₛᵃ = E[Rₜ₊₁ | Sₜ = s, Aₜ = a]
o	γ: diszkontálási tényező, γ ∈ [0, 1]
-	A különbség az MRP-hez képest: az MDP-ben van cselekvés (A), és mind az átmenet-valószínűségek (P), mind a jutalom (R) függ a választott cselekvéstől is. Az ágens dönthet, hogy mit cselekszik, és ez befolyásolja a jövőt.
Policy
-	A π politika a cselekvések eloszlása az állapotok függvényében:
o	π(a|s)=P[Aₜ=a|Sₜ=s]
-	Ez megadja annak valószínűségét, hogy az a cselekvést választjuk, ha az s állapotban vagyunk.
-	Fontos tulajdonságok:
o	A politika teljesen meghatározza az ágens viselkedését
o	MDP-ben a politika csak az aktuális állapottól függ (nem a teljes történettől)
o	A politikák stacionáriusak (időfüggetlenek) – ugyanabban az állapotban mindig ugyanazt az eloszlást követjük, függetlenül attól, hogy mikor érkeztünk oda
-	Ha adott egy M = (S, A, P, R, γ) MDP és egy π politika:
o	Az állapotsorozat S₁, S₂, ... egy Markov-folyamat (S, Pπ)
o	Az állapot és jutalom sorozat S₁, R₂, S₂, ... egy Markov jutalom-folyamat (S, Pπ, Rπ, γ), ahol:
 
	tehát például annak valószínűsége, hogy egy adott policy szerint a s-ből s’-be megyünk át egyenlő annak összesített valószínűségével, hogy az adott policy szerint s állapotban a cselekvést választjuk szorozva annak valószínűségével, hogy az a cselekvés s-ből s’-be visz át
State-value function
-	vπ(s) = Eπ[Gₜ|Sₜ=s]
-	Ez a várható hozam az s állapotból indulva, majd a π politikát követve. Megmondja, mennyire jó egy állapotban lenni az adott politika mellett.
-	vπ(s): „Mennyire jó ebben az állapotban lenni az adott policyt követve?"
Action-value function
-	qπ(s,a) = Eπ[Gₜ|Sₜ=s,Aₜ=a]
-	Ez a várható hozam az s állapotból indulva, az a cselekvést végrehajtva, majd a π politikát követve. Megmondja, mennyire jó egy adott cselekvést végrehajtani egy adott állapotban.
-	qπ (s, a): „Mennyire jó ebben az állapotban ezt a cselekvést választani a policit követve?"
-	A q-függvény részletesebb információt ad, mert cselekvésenként bontja le az értéket.
Belman Expectation Equation
-	The state-value function can again be decomposed into immediate reward plus discounted value of successor state:
 
-	The action-value function can similarly be decomposed,
 
 
 
 
 
-	mátrix formában:
 
 
Optimális értékfüggvény
-	Optimális állapot-értékfüggvény
o	v*(s)=maxπvπ(s)
o	Ez a maximális értékfüggvény az összes lehetséges politika közül – a legjobb elérhető érték az s állapotban.
 
-	Optimális cselekvés-értékfüggvény
o	q*(s, a) = maxπqπ(s,a)
o	Ez a maximális cselekvés-értékfüggvény az összes politika közül – a legjobb elérhető érték, ha az s állapotban az a cselekvést választjuk.
 
-	Az optimális értékfüggvény megadja a legjobb lehetséges teljesítményt az MDP-ben
-	Egy MDP-t „megoldottnak" tekintünk, ha ismerjük az optimális értékfüggvényt
-	Ebből közvetlenül származtatható az optimális politika
-	Részleges rendezés a politikákon
o	π≥π' ha vπ(s)≥vπ'(s) minden s állapotra
-	Tétel – Minden Markov döntési folyamatra:
o	Létezik optimális politika π*, amely jobb vagy egyenlő az összes többi politikánál: π*≥π minden π-re
o	Minden optimális politika eléri az optimális állapot-értékfüggvényt:
	vπ*(s) = v*(s)
o	Minden optimális politika eléri az optimális cselekvés-értékfüggvényt:
	qπ*(s, a) = q*(s, a)
o	Mindig létezik legalább egy determinisztikus optimális politika, és ha ismerjük q*(s, a)-t, akkor az optimális politika egyszerűen: minden állapotban válaszd azt a cselekvést, amelyre q*(s, a) maximális.
-	Ha ismerjük q*(s, a)-t, az optimális politika egyszerűen megkapható:
 
o	Vagyis: minden állapotban válaszd azt a cselekvést, amelyre q*(s, a) a legnagyobb.
o	Minden MDP-hez mindig létezik determinisztikus optimális politika (nem kell sztochasztikusan választani)
o	Ha ismerjük q*(s, a)-t, azonnal megvan az optimális politika – csak a maximumot kell választani
A Bellman-optimalitási egyenlet v*-ra
-	Az optimális értékfüggvények rekurzívan kapcsolódnak egymáshoz:
 
 
A Bellman-optimalitási egyenlet q*-ra
 
 

A Bellman-optimalitási egyenlet megoldása
-	A Bellman-optimalitási egyenlet nemlineáris (a max művelet miatt), ezért:
o	Nincs zárt formájú megoldás (általában)
o	Nem lehet egyszerűen mátrix-inverzióval megoldani, mint a Bellman-elvárási egyenletet
-	Iteratív megoldási módszerek:
o	Value Iteration (érték-iteráció)
o	Policy Iteration (politika-iteráció)
o	Q-learning
o	SARSA
o	Ezek a módszerek fokozatosan közelítik az optimális értékfüggvényt vagy politikát, amíg konvergálnak a megoldáshoz.
 
MINTA PÉLDA