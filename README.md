[//]: # (Image References)
[image1]: ./assets/YOLO_Architecture_from_the_original_paper.png "yolo_architect" 
[image2]: ./assets/pytesseract_flow.png "pystesseract_architect" 

4 pont: adatbázis feldolgozása, state of the art technologiak, valtozatok, megvalositas, eredmenyek teszteles

# Az algoritmus felépítése
A rendszámfelismerő algoritmust 4 érdemi részfeladatra bontottuk:
1. Az első rész során célunk a rendszámtábla minél pontosabb detektálása volt az adott képen.
2. A második részfeladat célja a kapott kép standard méretűre skálázása, és a rajta levő karakterek minél pontosabb szűrése, zajok elnyomása különböző képfeldolgozási módszerekkel, morfológiai eljárásokkal.
3. A rendszámot alkotó karakterek felismerése a standardizált képen.
A 4. részfeladat az előző három részfeladat megfelelő összeköttetése és kommunikációja (ide értve a hibakezelést is), illetve az adatbázisból a gépjárművek képeinek kinyerése, és azok feldolgozásának ciklusba rendezése.

Ehhez a felépítéshez a projekt során végig ragaszkodtunk, azonban az egyes lépésekhez többféle algoritmust is kipróbáltunk, bízva a robusztusság növekedésében.
1. részfeladat:

A rendszámtábla detektáláshoz a YOLO (You Only Look Once) objektumdetektáló state-of-the-art algoritmust használtuk. Ennek alkalmazási lehetősége széleskörű, főképp valós idejű objektumdetektálásra használják. Az eljárást 3 fő szempont miatt találtuk megfelelőnek:
* Egyrészt a sebessége igen jónak számít, 45 képkockát képes feldolgozni 1 másodperc alatt.
* Nagy pontossággal detektál, a hátteret és az objektumot jól szétválasztja - ezt a projekt során magunk is tapasztaltuk.
* Jó az általánosító képessége, ami a rendszámtáblák esetén különösen fontos, hiszen kevés konkrét közös jellemzője van a rendszámtábláknak, az információ jelentős hányada maguk a rajta szereplő karakterek. 

## YOLO modul
Az algoritmus összesen 24 konvolúciós rétegből áll, melyből négy réteg max-pooling réteg, illetve két réteg teljesen összekapcsolt réteg. A teljes architechtúrát tekintve egy teljesen kapcsolódó neurális hálóhóz (FCNN) hasonlít leginkább, amely alább látható.

  ![alt text][image1]
  *<center>[a YOLO modul architektúrája][1]</center>*

Az alkalmazott aktiválási függvény ReLU, kivéve az utolsó réteget, amely lineáris aktiválási függvényt használ.
A modell az objektumdetektálási problémát osztályozási feladat helyett regressziós feladatként kezeli meg azáltal, hogy határoló dobozokat (bounding box) különít el, és egyetlen konvolúciós neurális hálózat (CNN) segítségével valószínűségeket rendelnek az egyes detektált dobozokhoz.
A YOLO ezeknek a határoló dobozoknak az attribútumait egyetlen regressziós modul segítségével határozza meg a következő formátumban, ahol Y az egyes határoló dobozok végső vektoros ábrázolása:
Y = [pc, bx, by, bh, bw, c1, c2]
A fenti notációban pc annak a valószínűségi pontszámnak felel meg, hogy a doboz tartalmaz egy objektumot. A bx, by, bh, bw a bounding box koordináta leíró paraméterei; a c1 és c2 pedig a konkrét objektumtípusok valószínűsége az adott boundung boxban (esetünkben egy c paraméter lesz, mivel egyféle rendszámtábla típusú objektumunk lesz).

Az algoritmus betanítása nagyjából 1300 db képen történt, amely elég hatékonynak bizonyult, a 

A YOLO algoritmus hatékonyságát tekintve megfelelőnek mutatkozott a probléma megoldására. A futtatás során nagyon ritkán ütköztünk abba a problémába, hogy egyet sem vagy pedig több rendszámtáblát talált az adott képen. Mindemellett a rendszámtáblát jól adta vissza, nem vágott le belőle karaktereket, és a hátteret is jól kiszűrte, riktán hagyott meg a képen a rendszámtáblán kívül más érdemi objektumot.

A részproblémák közül a második jelentette a legnagyobb feladatot, mivel a kapott adatbázisban a képek a rendszámtábla olvashatóságát tekintve nagy szórást mutattak. A YOLO által már kiemelt képeken többféle képtranszformációval is próbálkoztunk. A második részfeladat sikeressége önmagában nehezen számszerűsíthető, mivel ennek eredménye csak a szövegfelismerés (3. részfeladat) elvégzése után látható, illetve nagyban függ a szövegfelismerő algoritmus jellegétől. A továbbiakban a tapasztalatainkat ennek tükrében foglaltuk össze.
A képfeldolgozás első lépése az egységes méretre történő skálázás, illetve a kép körbevágása annak érdekében, hogy a rendszámtáblákből csak a karakterek látszódjanak. Ezt követően elsőként a teljes kép binarizálását próbáltuk ki, azonban a szöveg visszaállítása nehézkesnek mutatkozott, a szövegfelismerő algoritmus rendre problémába ütközött a binarizált képpel: sokszor ismert fel rosszul karaktereket, és egyes karaktereket pedig nem azonosított karakterként. Ezt követően a kép szürkeárnyalatossá tételével probálkoztunk, ami robusztusabbnak mutatkozott a binarizálásnál, így a továbbiakban ezt a lépést meghagytuk. A tapasztalataink alapján az ezt követően végrehajtott Otsu-féle adaptív binarizálás javította a karakterfelismerés hatékonyságát, így a szürkeárnyalatos képen alkalmaztuk ezt az eljárást. A zajszűrés érdekében különböző méretű (3x3, 5x5, 9x9, 12x12) kernelekkel is végeztünk Gauss-simítást a képeken, azonban ezzel nem értünk el javulást.
Hibaként felmerült, hogy a szövegfelismerő algoritmus perspektivikusan torzított bemenetet kap, és vélhetően emiatt hibázik. Ennek érdekében a következőképp jártunk el:
elsőként egy általános perspektív transzformációt hajtunk végre annak érdekében, hogy a rendszámtáblák közel egységesek legyenek, azaz ne legyenek egyik irányba sem jelentősen torzítottak.
ezt követően betanítottunk egy karaktert, mint objektumot felismerő YOLO algoritmust, ezzel detektálva a rendszámtáblán az első, illetve az utolsó karakter koordinátáit. Ennek segítségével pedig már egy pontosabb perspektív transzformációt hajtunk végre az első lépés végén kapott képen. A két transzformációt azért tartottuk szükségesnek, mivel így a második YOLO modell precízebbnek mutatkozott, nagy hatékonysággal adta meg a koordinátákat.
Amennyiben a modell nem találta meg a karaktereket, úgy a szürkeárnyalatos képpel dolgoztunk tovább. Amennyiben megtalálta, úgy a transzformált képen három további transzformációt hajtottunk végre:
először a képhez hozzáadtunk egy fehér keretet, ami javította a szövegfelismerő modul hatékonyságát (enélkül nehezen ismerte fel a kép szélén lévő karaktereket)
utána pedig egy dilatáció, majd egy nyitás morfológiai műveletet hajtottunk végre.

A fenti eljárások sorozata már egy hatékony, a képek tulajdonságaira nézve (fény-árnyék hatások, képkészítés szöge a rendszámtáblához képest, rendszámtábla nagysága) robusztusnak bizonyult, így a szövegfelismerést ezen eljárásokat követően hajtottuk végre.

A harmadik részprobléma maga a karakterek felismerése volt a kiemelt képen. Ehhez a pytesseract state-of-the-art optikai karakterfelismerő (OCR) modult használtuk. A modell már betanított formában áll rendelkezésre telepítés után, így annak betanítását nem kellett elvégeznünk. A modell elviekben az alábbi preprocesszálási műveleteket hajtja végre a karakterfelismerés előtt ebben a sorrendben:
-invertálás
-átméretezés
-binarizáció
-zajszűrés
-dilatáció és erózió
-forgatás
A modell teljes működését az alábbi folyamatábra szemlélteti: 
  - a Pytesseract modul architektúrája  
  ![alt text][image2]

Mivel a kimenetként gyakran nem rendszámtábla típusú stringet kaptunk, így az OCR modul eredményének javítása érdekében felhasználtuk azt a tényt, hogy rendszámtáblák a bemenetek, azaz:
-egyrészt a kimenetben szereplő karakterek mind alfanumerikusak,
-másrészt a rendszámtábla első három karaktere mindenképp betű, az utolsó három karaktere pedig mindenképp szám.
Ebből adódóan definiáltunk egy whitelist-et az alfanumerikus karakterekből, mint lehetséges kimeneti értékek, illetőleg a hasonló alfanumerikus karakterek (pl. 2 és Z, 5 és S) cseréjét végeztük el annak helye alapján.
Egy másik probléma, amely fellépett a tesztelés során, hogy az OCR modul gyakran nem a megfelelő számú karaktert ismerte fel. Ennek megoldására a következőt próbáltuk ki: a már kiemelt képeken is betanítottunk egy YOLO modult annak érdekében, hogy a már kiemelt képet további képekre bontsa, amin egyenként láthatóak a karakterek. Ez azonban nem bizonyult túl hatékonynak a gyakorlatban, továbbá problémát jelentett egyszerre három YOLO modul futtatása, így ezt az opciót elvetettük.

Utolsó feladatként a három már ismertett szubrutin összehangolását végeztük el egy main.py fájlba, illetve a futtatáshoz szükséges adatokat nyertük ki az adatbázisból. Az olvashatóság végett a programkód megírásakor figyeltünk az OOP programszervezési elvre. A hivatkozások megnyitásához a urllib.request könyvtárat importáltuk, a beolvasott képeken elvégzett keresés eredményeit pedig egy *.csv fájlba töltöttük vissza.

# Hivatkozások
[1]: <https://arxiv.org/pdf/1506.02640.pdf> "pool"