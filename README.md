[//]: # (Image References)
[image1]: ./assets/BME_logo.jpg "BME"
[image2]: ./assets/YOLO_Architecture_from_the_original_paper.png "yolo_architect" 
[image3]: ./assets/pytesseract_ocr_flowchart.png "pystesseract_ocr_flowchart.png"
[image4]: ./assets/pytesseract_flowchart.png "pystesseract_flowchart.png"
[image5]: ./assets/main_flowchart.drawio.png "pystesseract_flowchart"
[image6]: ./assets/preprocessing_flowchart.drawio.png "pystesseract_flowchart"
[image7]: ./assets/plate_examples1.png "plate_examples1"
[image8]: ./assets/plate_examples2.png "plate_examples2"


[comment]: <> (főbb vázlatpontok: bevezető; state-of-the-art technológiák; algoritmus megvalósítása az egyes részváltozatokkal együtt; eredmények, tesztelés; továbbfejlesztési lehetőségek;hivatkozások)

<br /><br /><br />

![alt text][image1]
*<center></center>*

<br /><br />
<center> 

<strong><font size="15">Gépi látás</font> <br /></strong>
<em><font size="5">BMEVIIIM021</font><br /></em>
<font size="5">2022-23/2. szemeszter</font>
        
<br /><br /><br />

<font size="5">
      Féléves projektfeladat:<br />
      Okosváros projekt - <br />automatikus rendszámtábla felismerés
</font>

<br /><br /><br />

<font size="3">
      Készítették:<br />
      Kazup Dániel <em>@KazupD</em><br />
      Kocziha Barnabás <em>@KoczihaB</em><br />
      Kovács Tamás <em>@kovszegtom</em><br />
      Petrőtei Tamás József <em>@petroteitamas</em><br />
</font>

</center>

<br /><br /><br /><br /><br /><br />

# Tartalomjegyzék
1. [Bevezető](#introduction)
2. [Alkalmazott state-of-the-art technológiák](#state-of-the-art)
    1. [YOLO modul](#YOLO)
    2. [Tesseract modul](#Tesseract)
3. [Az algoritmus általános felépítése](#architect)
    1. [részfeladat](#first_subtask)
    2. [részfeladat](#second_subtask)
    3. [részfeladat](#third_subtask)
    4. [részfeladat](#fourth_subtask)
4. [Eredmények, tesztelés, konklúzió](#results)
5. [Továbbfejlesztési lehetőségek](#ideas)
6. [Hivatkozások, felhasznált források](#references)

<br /><br /><br /><br /><br /><br />

# 1. Bevezető <a name="introduction"></a>
A Gépi látás tárgyhoz kapcsolódóan a fakultatív házi feladat keretében egy okosváros projektet valósítottunk meg: rendszámfelismerő algoritmust fejlesztettünk. A problémát izgalmasnak találtuk, mivel a félév során alkalmazott technológiák, módszereket valós adatokon is kipróbálhattuk, azok hatékonyságát tesztelhettük, így például a morfológiai műveleteket, binarizálást, zajszűrést vagy éppen deep learning technológiát. Az elkészült algoritmus felépítését, működését és az eredményeket az alábbiakban ismertetjük.

# 2. Alkalmazott state-of-the-art technológiák <a name="state-of-the-art"></a>

## YOLO modul <a name="YOLO"></a>
Az algoritmus összesen 24 konvolúciós rétegből áll, melyből négy réteg max-pooling réteg, illetve két réteg teljesen összekapcsolt réteg. A teljes architechtúrát tekintve egy teljesen kapcsolódó neurális hálóhóz (FCNN) hasonlít leginkább, amely alább látható.

  ![alt text][image2]
  *<center>[1. kép: a YOLO modul architektúrája][1]</center>*

Az alkalmazott aktiválási függvény ReLU, kivéve az utolsó réteget, amely lineáris aktiválási függvényt használ.
A modell az objektumdetektálási problémát osztályozási feladat helyett regressziós feladatként kezeli meg azáltal, hogy határoló dobozokat (bounding box) különít el, és egyetlen konvolúciós neurális hálózat (CNN) segítségével valószínűségeket rendelnek az egyes detektált dobozokhoz.
A YOLO ezeknek a határoló dobozoknak az attribútumait egyetlen regressziós modul segítségével határozza meg a következő formátumban, ahol Y az egyes határoló dobozok végső vektoros ábrázolása:
Y = [pc, bx, by, bh, bw, c1, c2]
A fenti notációban pc annak a valószínűségi pontszámnak felel meg, hogy a doboz tartalmaz egy objektumot. A bx, by, bh, bw a bounding box koordináta leíró paraméterei; a c1 és c2 pedig a konkrét objektumtípusok valószínűsége az adott boundung boxban (esetünkben egy c paraméter lesz, mivel egyféle rendszámtábla típusú objektumunk lesz).

## Tesseract modul <a name="Tesseract"></a>

A Tesseract vagy Pytesseract könyvtár egy Hewlett-Packard Co. által, majd későbbiekben a Google által fejlesztett ún. *OCR (Optical Character Recognition)* open-source motor. A felhasználása sokoldalú, a Pytesseract segítségével könnyedén implementálhatók OCR megoldások például képfeldolgozási projektekben, dokumentumok digitalizálásában vagy adatelemzésben.
Az általános karakterfelismerés a Tesseract modul esetében a következőképpen működik:

  ![alt text][image3]
  *<center>[2. kép: a Tesseract modul architektúrája][2]</center>*

A könyvtár telepítés után további tanítás nélkül használható. A karakterfelismerést megelőzően elviekben az alábbi preprocesszálási műveleteket hajtja végre a karakterfelismerés előtt ebben a sorrendben:
- invertálás
- átméretezés
- binarizáció
- zajszűrés
- dilatáció és erózió
- forgatás

A tesztelés során azonban gyakran ütköztünk abba a problémába, hogy a modul nem vagy nem megeflelően ismerte fel a képen látható karaktereket, így ezek műveletek javarészét magunk is implementáltuk a saját preprocesszálási szubrutinunkba a magasabb hatékonyság elérése érdekében.

A modell belső működését az alábbi folyamatábra szemlélteti:

  ![alt text][image4]
  *<center>[3. kép: a Tesseract modul architektúrája][3]</center>*


# 3. Az algoritmus általános felépítése <a name="architect"></a>
A rendszámfelismerő algoritmust négy érdemi részfeladatra bontottuk:
1. Az első rész során célunk a rendszámtábla objektum minél pontosabb detektálása, kiemelése volt a kép többi, számunkra érdemi információt nem tartalmazó háttértől. Ennek megvalósítását a **detect_plate.py** fájlban, illetve az ehhez szükséges függvényeket az azonos nevű classban implementáltuk.
2. A második részfeladat célja a kapott kép preprocesszálása, azaz standard méretűre skálázása, és a rajta levő karakterek minél pontosabb szűrése, zajok elnyomása különböző képfeldolgozási módszerekkel, morfológiai eljárásokkal. Ennek megvalósítása a **detect_plate.py** class **process_image** függvényében történt.
3. A rendszámot alkotó karakterek felismerése a standardizált képen, a rendszámot leíró string előállítása. Az ehhez tartozó megvalósítás az **image_to_text.py** fájlban található.

A 4. részfeladatként az előző három részfeladat megfelelő összeköttetését és kommunikációját (ide értve a hibakezelést is), illetve az adatbázisból a gépjárművek képeinek kinyerését, és azok feldolgozásának ciklusba rendezését határoztuk meg. Az adatbázis kezelése a **fetch_car.py** fájlban, és az azonos nevű osztályban került implementálásra.

Az algoritmus teljes működését a következő ábrán szemléltettük:

  ![alt text][image5]
  *<center>4. kép: a teljes algoritmus architektúrája</center>*

Ehhez a felépítéshez a projekt során végig ragaszkodtunk, azonban az egyes lépésekhez többféle algoritmust is kipróbáltunk, növelve az algoritmus robusztusságát a bemeneti képeket terhelő zajokra, torzításokra, elmosódásokra nézve.

## 1. részfeladat: <a name="first_subtask"></a>

A rendszámtábla detektáláshoz a YOLO (You Only Look Once) objektumdetektáló state-of-the-art algoritmust használtuk. Ennek alkalmazási lehetősége széleskörű, főképp valós idejű objektumdetektálásra használják. Az eljárást 3 fő szempont miatt találtuk megfelelőnek:
* Egyrészt a sebessége igen jónak számít, 45 képkockát képes feldolgozni 1 másodperc alatt.
* Nagy pontossággal detektál, a hátteret és az objektumot jól szétválasztja - ezt a projekt során magunk is tapasztaltuk.
* Jó az általánosító képessége, ami a rendszámtáblák esetén különösen fontos, hiszen kevés konkrét közös jellemzője van a rendszámtábláknak, az információ jelentős hányada maguk a rajta szereplő karakterek.

Az algoritmus betanítása 90 db adaton történt, amely elegendő számúnak bizonyult a tesztelés során. A betanítás eredményét a **trained_90.pt** fájlban tároltuk el. (A tanítást elvégeztük 400 adapontra is, azonban ekkora mintán a betanítás paradox módon kevésbé mutatkozott optimálisnak.) A YOLO algoritmus hatékonyságát tekintve megfelelőnek mutatkozott a probléma megoldására, a futtatás során nagyon ritkán ütköztünk abba a problémába, hogy egyet sem vagy pedig több rendszámtáblát talált az adott képen. Mindemellett a rendszámtáblát jól adta vissza, nem vágott le belőle karaktereket, és a hátteret is jól kiszűrte, riktán hagyott meg a képen a rendszámtáblán kívül más érdemi objektumot.

Az első részfeladatot megvalósító **detect_plate** class annak főbb függvényeivel:
+ ***get_plate_image(self, image):*** A rendszámdetektálásra betanított YOLO modul által történő rendszámkiemelés (kimenetként a bounding box két sarkának koordinátáit adja vissza).
+ ***process_image(self, image):*** A függvény elvégzi a szükséges preprocesszálási műveleteket a kiemelt képen, kimenetként a transzformált, kiemelt képet adja vissza.
+ ***perspective_correction(self, image, general_resize_factor):*** A függvény kifeszíti a bemenetként adott képet a megfelelő arányra. 

## 2. részfeladat: <a name="second_subtask"></a>

A részproblémák közül a második jelentette a legnagyobb feladatot, mivel a kapott adatbázisban a képek a rendszámtábla olvashatóságát, minőségüket tekintve nagy szórást mutattak. A YOLO által már kiemelt képeken többféle képtranszformációval is próbálkoztunk. A második részfeladat sikeressége önmagában nehezen számszerűsíthető, mivel ennek eredménye csak a szövegfelismerés, azaz a 3. részfeladat elvégzését követően értelmezhető, illetve nagyban függ a szövegfelismerő algoritmus jellegétől. A továbbiakban a tapasztalatainkat ennek tükrében foglaltuk össze.

A képfeldolgozás első lépése az egységes méretre történő skálázás, illetve a kép körbevágása annak érdekében, hogy a rendszámtáblákből csak a karakterek látszódjanak. Ezt követően elsőként a teljes kép binarizálását próbáltuk ki, azonban a szöveg visszaállítása nehézkesnek mutatkozott, a szövegfelismerő algoritmus rendre problémába ütközött a binarizált képpel: sokszor ismert fel rosszul karaktereket, és egyes karaktereket pedig nem azonosított karakterként. Ezt követően a kép szürkeárnyalatossá tételével probálkoztunk, ami robusztusabbnak mutatkozott a binarizálásnál, így a továbbiakban ezt a lépést meghagytuk. A tapasztalataink alapján az ezt követően végrehajtott Otsu-féle adaptív binarizálás javította a karakterfelismerés hatékonyságát, így a szürkeárnyalatos képen alkalmaztuk ezt az eljárást. 

A zajszűrés érdekében különböző méretű (3x3, 5x5, 9x9, 12x12) kernelekkel is végeztünk Gauss-simítást a képeken, azonban ezzel nem értünk el javulást.

Hibaként felmerült, hogy a szövegfelismerő algoritmus perspektivikusan torzított bemenetet kap, és vélhetően emiatt hibázik. Ennek érdekében a következőképp jártunk el:
elsőként egy általános perspektív transzformációt hajtunk végre annak érdekében, hogy a rendszámtáblák közel egységesek legyenek, azaz ne legyenek egyik irányba sem jelentősen torzítottak.
ezt követően betanítottunk egy karaktert, mint objektumot felismerő YOLO algoritmust, ezzel detektálva a rendszámtáblán az első, illetve az utolsó karakter koordinátáit. A betanítást 250 adaton végeztük, a betanítás eredményét a **yolo_custom_char_250.pt** fájlban tároltuk el. Ennek segítségével pedig már egy pontosabb perspektív transzformációt hajtunk végre az első lépés végén kapott képen. A két transzformációt azért tartottuk szükségesnek, mivel így a második YOLO modell precízebbnek mutatkozott, nagy hatékonysággal adta meg a koordinátákat.

Amennyiben a modell nem találta meg a karaktereket, úgy a szürkeárnyalatos képpel dolgoztunk tovább. Amennyiben megtalálta, úgy a transzformált képen három további transzformációt hajtottunk végre:
először a képhez hozzáadtunk egy fehér keretet, ami javította a szövegfelismerő modul hatékonyságát (enélkül nehezen ismerte fel a kép szélén lévő karaktereket)
utána pedig egy dilatáció, majd egy nyitás morfológiai műveletet hajtottunk végre. A kiemelt képen végzett preprocesszálási műveleteket az alábbi folyamatábrán foglaltuk össze:

  ![alt text][image6]
  *<center>5. kép: a preprocesszálást megvalósító algoritmus</center>*

A fenti eljárások sorozata már egy hatékony, a képek tulajdonságaira nézve (fény-árnyék hatások, képkészítés szöge a rendszámtáblához képest, rendszámtábla nagysága) robusztusnak bizonyult, így a szövegfelismerést ezen eljárásokat követően hajtottuk végre. A teljes preprocesszálást a **detect_plate** class **process_image(self, image)** függvénye valósítja meg.

## 3. részfeladat: <a name="third_subtask"></a>

A harmadik részprobléma maga a karakterek felismerése volt a kiemelt képen. Ehhez a pytesseract state-of-the-art optikai karakterfelismerő (OCR) modult használtuk. A modell már betanított formában áll rendelkezésre telepítés után, így annak betanítását nem kellett elvégeznünk.

Mivel a kimenetként gyakran nem rendszámtábla típusú stringet kaptunk, így az OCR modul eredményének javítása érdekében felhasználtuk azt a tényt, hogy rendszámtáblák a bemenetek, azaz:
* egyrészt a kimenetben szereplő karakterek mind alfanumerikusak,
* másrészt a rendszámtábla első három karaktere mindenképp betű, az utolsó három karaktere pedig mindenképp szám.

Ebből adódóan definiáltunk egy whitelist-et az alfanumerikus karakterekből, mint lehetséges kimeneti értékek, illetőleg a hasonló alfanumerikus karakterek (pl. 2 és Z, 5 és S) cseréjét végeztük el RegEx műveletekkel a karakterek helye alapján.

Egy másik probléma, amely fellépett a tesztelés során, hogy az OCR modul gyakran nem a megfelelő számú karaktert ismerte fel. Ennek megoldására a következőt próbáltuk ki: a már kiemelt képeken is betanítottunk egy YOLO modult annak érdekében, hogy a már kiemelt képet további képekre bontsa, amin egyenként láthatóak a karakterek. Ez azonban nem bizonyult túl hatékonynak a gyakorlatban, továbbá problémát jelentett egyszerre három YOLO modul futtatása, így ezt az opciót elvetettük.

A részfeladatot megvalósító **image_to_text** classban implementált fontosabb függvények:
+ ***get_text(self, image):*** A Tesseract OCR modul visszaadja a bemeneti képen található stringet.
+ ***format_text(self, text):*** A string korrekciója RegEx műveletek segítségével.

## 4. részfeladat: <a name="fourth_subtask"></a>

Utolsó feladatként a három már ismertett szubrutin összehangolását végeztük el egy main.py fájlba, illetve a futtatáshoz szükséges adatokat nyertük ki az adatbázisból. Az olvashatóság végett a programkód megírásakor figyeltünk az OOP programszervezési elvre. A hivatkozások megnyitásához a urllib.request könyvtárat importáltuk, a beolvasott képeken elvégzett keresés eredményeit pedig egy *.csv fájlba töltöttük vissza.

A hibakezelést az egyes részfeladatok során valósítottuk meg, ami az alább részletezett hibák kezelését takarta.
* A kapott adatbázisban található URL olyan weboldalra mutatott, ami 404-es hibát eredményezett.
* A YOLO teljes rendszámtábla detektálásra betanított modellje esetenként nem talált rendszámtáblát, így nem adott vissza *bounding box* koordinátákat.
* A YOLO karakterek detektálására betanított modellje egyes esetekben nem találta meg a karaktereket, így nem adott vissza *bounding box* koordinátákat. Megoldásként ebben az esetben a teljes rendszámtábla szürkeárnyalatos képét használtuk a Tesseract modulban.
* A Tesseract nem találta a kapott bemeneten a karaktereket.

A **fetch_car.py** fájlban definiált, az adatbázis kinyeréséhez használt fontosabb függvények azok leírásával és az argumentumaiknak felsorolásával:

+ ***load_by_numberplate(self, numberplate):*** A bemenetként megadott *numberplate* stringhez tartozó *képeket* adja vissza. A szubrutin a tanítási adatbázison értelmezett.
+ ***load_by_index(self, index):*** A bemenetként megadott *index* változóhoz tartozó *képeket* adja vissza. A szubrutin a tanítási adatbázison értelmezett.
+ ***get_index_by_numberplate(self, numberplate):** A bemenetként megadott *numberplate* stringhez tartozó *indexet* adja vissza.
+ ***get_numberplate_by_index(self, index):*** Kimenetként a megadott *indexhez* tartozó *numberplate* stringet adja vissza.
+ ***load_image_by_url(self, url):*** A megadott *URL*-en lévő *képet* adja vissza kimenetként.

Az átláthatóbb működés érdekében a teljes algoritmus egy adott adatbázisra a **main.py** fájlon belül definiált függvényekkel hivható meg:

+ ***test_on_database(img_fetch, img_plate_detect, img_to_text)***: Teszteléshez használt függvény, csupán az adatbázis egy részén végzi el a rendszámdetektálást.
+ ***test_on_final_database(img_fetch, img_plate_detect, img_to_text)***: A függvény a teljes adatbázison elvégzi a rendszámdetektálást.

# 4. Eredmények, tesztelés, konklúzió <a name="results"></a>
[comment]: <> (TODO)
Az algoritmus hatékonyságát folyamatosan teszteltük a munkafolyamat során, és igyekeztünk új szubrutinokkal hatékonyabbá tenni azt, így iteratív módon készítettük el a projektfeladatot. 

[comment]: <> (a 10-15% valid?)

Az első verzió hatékonysága a kiemelt kép minimálisan szükséges preprocesszálásával körülbelül 10-15%  volt. A binarizálás, illetve zajszűrési technikák implementálását követően is 40% alatt maradt a felismerési hatékonyság. A perspektíva korrekció, illetve a padding hozzáadásával azonban már szignifikánsan növekedett a hatékonyság: nagyjából 70%-nak bizonyult 1000 képből álló mintán.

A tapasztalataink alapján jellemzően a nagyon torzult képeken tipikusan szuboptimálisnak bizonyult az algoritmus, illetőleg az árnyékhatások is rontanak a hatékonyságon. Három tipikus hibát tapasztaltunk a tesztelés során:
* gyakran azonosított egy karaktert két másik karakterként (mindamellett, hogy a többi karaktert a rendszámtáblán jól felismerte), illetve
* egy karaktert szimplán másik karakternek érzékelt, továbbá
* a rendszám eleji "I" karaktert rendkívül kis hatékonysággal ismerte fel az OCR modul.

Az utóbbi probléma más karakterek elején is fennállt, azonban azok esetében megoldást jelentett a már ismertett fehér keret (padding) hozzáadása a kiemelt kép preprocesszálása során. Az alábbiakban néhány példát ismertetünk az algoritmus működésére vonatkozóan.

![alt text][image7]
  *<center>6. kép: néhány példa a bemenetből (hi-res) az OCR modulba adott képpel (preprocesszálás után)</center>*

A fenti példák esetében az algoritmus a következő eredményt produkálta:
* a bal oldali képeken nem ismerte fel a rendszámokat, de különböző okok miatt: a felső képen túl éles szögből készült, így a perspektíva korrekció sem tudta kellően kifeszíteni a képet; az alsó kép esetében pedig a preprocesszálás során kevésbé kontrasztosabbá vált a kép - és habár szabad szemmel jól látható a rendszám - az OCR mégsem ismerte fel, illetve
* a jobb oldali képeken az algoritmus helyesen ismerte fel a rendszámokat annak ellenére, hogy a felső képen kissé ferde, az alsó képen pedig éles szögből készült a kép.

A részletezett előforduló hibák kapcsán próbálkoztunk a második YOLO által felismert egyes karaktereket betáplálni az OCR algoritmusba, de a felismerés hatékonyága 20% alatt maradt, így ezt a módszert elvetettük.

![alt text][image8]
*<center>7. kép: további példák a bemenetből (hi-res)az OCR modulba adott képpel (preprocesszálás után)</center>*

A bal oldali kép esetében szintén nem járt sikerrel az algoritmus, mivel az eredeti kép túlexponáltnak bizonyult, amin a preprocesszálási eljárások sem tudtak érdemben javítani. A jobb oldali kép esetében pedig a preprocesszálás sikeresnek mondható volt, azonban a Tesseract OCR modul a "J" karaktert "U" karakterként azonosította, így nem ismerte fel sikeresen a rendszámot.

A képek homályossága meglepő módon kevésbé befolyásolta a felismerés hatékonyságát, ez főleg olyan esetekben mutatkozott problémásnak, amikor a kép nem volt kellően kontrasztos.

Az algoritmus számításigény szempontjából más szakirodalmi algoritmusokhoz képest nem mutatkozott szignifikánsan számításigényesebbnek: a fenti 1000 adatpontból álló bemenet mellett 740 [sec] alatt futott le, ami képenként átlagosan 0,74 [s] futási időt jelentett.


# 5. Továbbfejlesztési lehetőségek <a name="ides"></a>
[comment]: <> (TODO)

Az algoritmus elsősorban a régi (hagyományos), egysoros, normál méretű, nem egyedi rendszámtábla felismerésére van illesztve. A kétsoros és az új formátumú rendszámtábla helyzetfelismerése megoldott, azonban a rajta lévő karakterek kiolvasása jelenleg problémás.

Mindemellett megfogalmaztunk néhány lehetséges továbbfejlesztési lehetőséget a tapasztalataink alapján a hagyományos rendszámtábla felismerés javítása érdekében:
- a perspektív transzformáció algoritmusának továbbfejlesztése,
- a rendszámtábla külső befolglaló éleinek detektálása,
- sarokkereső algoritmus használata,
- OCR algoritmusának továbbfejlesztése (mivel a szűk keresztmetszetnek az OCR robusztussága bizonyult),
- a rendszámtábla betűtípusának feltanítása egy már meglévő OCR hálóra (transfer learning), illetve
- az algoritmus optimalizálása GPU-ra, ezáltal a számítási sebesség növelhető és az erőforrásigény csökkenthető.

# 6. Hivatkozások, felhasznált források <a name="references"></a>
1. https://arxiv.org/pdf/1506.02640.pdf
2. https://nanonets.com/blog/ocr-with-tesseract/
3. https://www.researchgate.net/figure/Tesseract-OCR-engine-process-8_fig2_341936500

Egyéb felhasznált források:
1. Ray Smith: An Overview of the Tesseract OCR Engine<br>
https://tesseract-ocr.github.io/docs/tesseracticdar2007.pdf
2. Tesseract GitHub main repository: <br>
https://github.com/tesseract-ocr/tesseract
3. https://www.datacamp.com/blog/yolo-object-detection-explained
4. https://www.section.io/engineering-education/introduction-to-yolo-algorithm-for-object-detection/
5. https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006
6. Lubna, Mufti N, Shah SAA. Automatic Number Plate Recognition:
A Detailed Survey of Relevant Algorithms. Sensors (Basel). 2021 Apr 26;21(9):3028. doi: 10.3390/s21093028.
(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8123416/)
7. https://viso.ai/computer-vision/automatic-number-plate-recognition-anpr/


[comment]: <> (a hivatkozott forrásokhoz hyperlink a hivatkozás helyén)
[1]: <https://arxiv.org/pdf/1506.02640.pdf> "yolo"
[2]: <https://nanonets.com/blog/ocr-with-tesseract/> "tesseract_general"
[3]: <https://www.researchgate.net/figure/Tesseract-OCR-engine-process-8_fig2_341936500> "tesseract_process"


[comment]: <> (credits goes to @KazupD, @KoczihaB, @petroteitamas, @kovszegtom)