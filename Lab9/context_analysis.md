# Context analysis — CoNLL-2003 test split

Các ví dụ dưới đây được trích từ gold spans của test split; chưa có phần giải thích thay sinh viên. Surface form được chuẩn hóa bằng `casefold()` và gộp whitespace để tìm trùng.

Số surface form có nhiều entity type: **69**. Đề yêu cầu ít nhất 3 trường hợp nếu test set cung cấp đủ.

| Surface form | Sentence A (id) | Type A | Sentence B (id) | Type B | Student analysis |
|---|---|---|---|---|---|
| china | SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . (0) | PER | But China saw their luck desert them in the second match of the group , crashing to a surprise 2-0 defeat to newcomers Uzbekistan . (4) | LOC | TODO |
| barcelona | FIFA 's players ' status committee , meeting in Barcelona , decided that although the Udinese document was basically valid , it could not be legally protected . (146) | LOC | Barcelona -LPR- Spain -RPR- 9 4 5 13 (179) | ORG | TODO |
| charleroi | Charleroi -LPR- Belgium -RPR- 9 0 9 9 (174) | ORG | In Charleroi : (855) | LOC | TODO |
| leicester | Barbarians - 15 - Tim Stimpson -LPR- England -RPR- ; 14 - Nigel Walker -LPR- Wales -RPR- , 13 - Allan Bateman -LPR- Wales -RPR- , 12 - Gregor Townsend -LPR- Scotland -RPR- , 11 - Tony Underwood -LPR- England -RPR- ; 10 - Rob Andrew -LPR- England -RPR- , 9 - Rob Howley -LPR- Wales -RPR- ; 8 - Scott Quinnell -LPR- Wales -RPR- , 7 - Neil Back -LPR- England -RPR- , 6 - Dale McIntosh -LPR- Pontypridd -RPR- , 5 - Ian Jones -LPR- New Zealand -RPR- , 4 - Craig Quinnell -LPR- Wales -RPR- , 3 - Darren Garforth -LPR- Leicester -RPR- , 2 - Norm Hewitt -LPR- New Zealand -RPR- , 1 - Nick Popplewell -LPR- Ireland -RPR- . (202) | LOC | Leicester 1 -LPR- Marshall 78 -RPR- Blackburn 1 -LPR- Sutton 34 -RPR- . (2878) | ORG | TODO |
| pontypridd | Barbarians - 15 - Tim Stimpson -LPR- England -RPR- ; 14 - Nigel Walker -LPR- Wales -RPR- , 13 - Allan Bateman -LPR- Wales -RPR- , 12 - Gregor Townsend -LPR- Scotland -RPR- , 11 - Tony Underwood -LPR- England -RPR- ; 10 - Rob Andrew -LPR- England -RPR- , 9 - Rob Howley -LPR- Wales -RPR- ; 8 - Scott Quinnell -LPR- Wales -RPR- , 7 - Neil Back -LPR- England -RPR- , 6 - Dale McIntosh -LPR- Pontypridd -RPR- , 5 - Ian Jones -LPR- New Zealand -RPR- , 4 - Craig Quinnell -LPR- Wales -RPR- , 3 - Darren Garforth -LPR- Leicester -RPR- , 2 - Norm Hewitt -LPR- New Zealand -RPR- , 1 - Nick Popplewell -LPR- Ireland -RPR- . (202) | LOC | Pontypridd 53 Bridgend 9 (2832) | ORG | TODO |
| tasmania | Score on the first day of the four-day Sheffield Shield match between Tasmania and Victoria at Bellerive Oval on Friday : (296) | LOC | Former Australia test batsman Dean Jones hit an unbeaten 130 to lead Victoria 's fightback in their Sheffield Shield match against Tasmania on Saturday . (3174) | ORG | TODO |
| victoria | Score on the first day of the four-day Sheffield Shield match between Tasmania and Victoria at Bellerive Oval on Friday : (296) | LOC | Tasmania 352 for three -LPR- David Boon 106 not out , Shaun Young 86 not out , Michael DiVenuto 119 -RPR- v Victoria . (297) | ORG | TODO |
| new york | NEW YORK 1996-12-06 (354) | LOC | NEW YORK 10 6 .625 3 (440) | ORG | TODO |
| league | League teams after games played on Thursday -LPR- tabulate under (356) | ORG | League game on Thursday -LPR- home team in CAPS -RPR- : (555) | LOC | TODO |
| hartford | HARTFORD 12 7 6 77 76 30 (361) | ORG | BALANCE - Hartford , Conn . (1630) | LOC | TODO |
| buffalo | BUFFALO 13 12 1 77 76 27 (362) | ORG | ANAHEIM AT BUFFALO (395) | LOC | TODO |
| boston | BOSTON 10 11 4 74 84 24 (363) | ORG | NEW JERSEY AT BOSTON (476) | LOC | TODO |
| montreal | MONTREAL 10 14 4 96 103 24 (364) | ORG | CHICAGO AT MONTREAL (3363) | LOC | TODO |
| pittsburgh | PITTSBURGH 9 13 3 81 91 21 (365) | ORG | SAN DIEGO AT PITTSBURGH (543) | LOC | TODO |
| ottawa | OTTAWA 7 11 6 62 72 20 (366) | ORG | EDMONTON 5 Ottawa 2 (3379) | LOC | TODO |
| florida | FLORIDA 17 4 6 83 53 40 (369) | ORG | 14 years later , Florida man dies for killing . (1659) | LOC | TODO |
| philadelphia | PHILADELPHIA 14 12 2 75 75 30 (370) | ORG | Philadelphia , which fell from an NFC East tie with the Dallas Cowboys and Washington Redskins , go on the road against the New York Jets and then entertain Arizona . (421) | LOC | TODO |
| new jersey | NEW JERSEY 14 10 1 61 61 29 (371) | ORG | DETROIT AT NEW JERSEY (3298) | LOC | TODO |
| washington | WASHINGTON 13 12 1 69 66 27 (372) | ORG | PITTSBURGH AT WASHINGTON (397) | LOC | TODO |
| ny islanders | NY ISLANDERS 7 11 8 65 72 22 (374) | ORG | WASHINGTON AT NY ISLANDERS (3362) | LOC | TODO |
| tampa bay | TAMPA BAY 8 15 2 69 81 18 (375) | ORG | WASHINGTON AT TAMPA BAY (544) | LOC | TODO |
| detroit | DETROIT 15 9 4 81 53 34 (379) | ORG | CLEVELAND AT DETROIT (477) | LOC | TODO |
| dallas | DALLAS 16 9 1 74 60 33 (380) | ORG | PHILADELPHIA AT DALLAS (399) | LOC | TODO |
| chicago | CHICAGO 12 12 3 71 67 27 (381) | ORG | MONTREAL AT CHICAGO (398) | LOC | TODO |
| st louis | ST LOUIS 13 14 0 78 81 26 (382) | ORG | St Louis 4 COLORADO 3 (3378) | LOC | TODO |
| toronto | TORONTO 11 15 0 76 89 22 (383) | ORG | " The negotiations will be held both in Toronto and in Jakarta , " one source , speaking on condition of anonymity , told Reuters . (2318) | LOC | TODO |
| colorado | COLORADO 17 6 4 97 56 38 (387) | ORG | ST LOUIS AT COLORADO (400) | LOC | TODO |
| vancouver | VANCOUVER 14 11 1 84 83 29 (388) | ORG | Vancouver 5 7 (1110) | LOC | TODO |
| edmonton | EDMONTON 13 14 1 94 88 27 (389) | ORG | OTTAWA AT EDMONTON (401) | LOC | TODO |
| los angeles | LOS ANGELES 11 13 3 72 83 25 (390) | ORG | -- David Brinkerhoff , Los Angeles bureau +1 213 380 2014 (2048) | LOC | TODO |
| san jose | SAN JOSE 10 13 4 69 87 24 (391) | ORG | TAMPA BAY AT SAN JOSE (3367) | LOC | TODO |
| indianapolis | INDIANAPOLIS 1996-12-06 (415) | LOC | INDIANAPOLIS 8 6 0 269 284 (495) | ORG | TODO |
| cincinnati | Indianapolis closes with games at Kansas City and Cincinnati . (419) | LOC | CINCINNATI 5 8 0 299 318 (503) | ORG | TODO |
| kansas city | Indianapolis closes with games at Kansas City and Cincinnati . (419) | LOC | KANSAS CITY 9 4 0 262 230 (508) | ORG | TODO |
| arizona | Philadelphia , which fell from an NFC East tie with the Dallas Cowboys and Washington Redskins , go on the road against the New York Jets and then entertain Arizona . (421) | ORG | DALLAS AT ARIZONA (545) | LOC | TODO |
| miami | MIAMI 14 4 .778 - (439) | ORG | NEW YORK AT MIAMI (478) | LOC | TODO |
| atlanta | ATLANTA 10 8 .556 7 (451) | ORG | On Thursday , the 11th Circuit U.S. Court of Appeals in Atlanta denied his appeal in federal court . (1671) | LOC | TODO |
| houston | HOUSTON 16 2 .889 - (459) | ORG | JACKSONVILLE AT HOUSTON (540) | LOC | TODO |
| utah | UTAH 14 2 .875 1 (460) | ORG | MINNESOTA AT UTAH (481) | LOC | TODO |
| denver | DENVER 5 14 .263 11 1/2 (463) | ORG | UTAH AT DENVER (3302) | LOC | TODO |
| san antonio | SAN ANTONIO 3 13 .188 12 (464) | ORG | VANCOUVER AT SAN ANTONIO (480) | LOC | TODO |
| seattle | SEATTLE 15 5 .750 - (468) | ORG | BUFFALO AT SEATTLE (547) | LOC | TODO |
| la lakers | LA LAKERS 13 7 .650 2 (469) | ORG | ORLANDO AT LA LAKERS (484) | LOC | TODO |
| portland | PORTLAND 11 8 .579 3 1/2 (470) | ORG | CHARLOTTE AT PORTLAND (482) | LOC | TODO |
| golden state | GOLDEN STATE 6 12 .333 8 (472) | ORG | INDIANA AT GOLDEN STATE (483) | LOC | TODO |
| sacramento | SACRAMENTO 6 12 .333 8 (473) | ORG | PHOENIX AT SACRAMENTO (479) | LOC | TODO |
| american | AMERICAN FOOTBALL CONFERENCE (490) | MISC | The conditions for clearance of the alliance were that British Airways and American drop 168 slots at London Heathrow airport , the busiest in Europe . (1016) | ORG | TODO |
| new england | NEW ENGLAND 9 4 0 355 269 (493) | ORG | NY JETS AT NEW ENGLAND (546) | LOC | TODO |
| oakland | OAKLAND 6 7 0 274 234 (510) | ORG | KANSAS CITY AT OAKLAND (551) | LOC | TODO |
| san francisco | SAN FRANCISCO 10 3 0 325 198 (529) | ORG | CAROLINA AT SAN FRANCISCO (548) | LOC | TODO |
| new orleans | NEW ORLEANS 2 11 0 184 291 (533) | ORG | ATLANTA AT NEW ORLEANS (542) | LOC | TODO |
| nebraska | Pace outdistanced three senior finalists -- Virginia Tech defensive end Cornell Brown , Arizona State offensive tackle Juan Roque and defensive end Jared Tomich of Nebraska . (563) | ORG | Nebraska fed cattle roundup - USDA . (1826) | LOC | TODO |
| freiburg | Karlsruhe 3 Freiburg 0 (596) | ORG | Karlsruhe 3 -LPR- Reich 29th , Carl 44th , Dundee 69th -RPR- Freiburg 0 . (679) | LOC | TODO |
| karlsruhe | Karlsruhe 3 Freiburg 0 (596) | ORG | Karlsruhe 3 -LPR- Reich 29th , Carl 44th , Dundee 69th -RPR- Freiburg 0 . (679) | LOC | TODO |
| milan | Weah served out his suspension during Milan 's 2-1 home defeat by Rosenborg of Norway on Wednesday . (808) | ORG | -- Milan newsroom +392 66129502 (1873) | LOC | TODO |
| czech | Lazio have injury doubts about striker Pierluigi Casiraghi , Czech midfielder Pavel Nedved and defender Paolo Negro , while Roma present a full strength side led by Argentine Abel Balbo , Marco Delvecchio and Francesco Totti in attack . (831) | LOC | HAVEL PRAISES CZECH NATIVE ALBRIGHT AS FRIEND . (889) | MISC | TODO |
| cardiff | Wales grappled with its own cluster of meningitis cases on a university campus in Cardiff . (1048) | LOC | Caerphilly 20 Cardiff 34 (2829) | ORG | TODO |
| nats | The contract involved upgrading the flight data processing system at the Oceanic Control Centre in Prestwick in south west Scotland for National Air Traffic Services Ltd -LPR- NATS -RPR- , subsidiary of the Civil Aviation Authority . (1068) | ORG | NZ 's Bolger says Nats to meet NZ First on Sunday . (2231) | PER | TODO |
| santa fe | Wall St speculates about Santa Fe savior . (1150) | LOC | Homestake Mining Co tops Wall Street 's list as the most likely white knight buyer for Santa Fe Pacific Gold Corp if Santa Fe rejects unsolicited suitor Newmont Mining Corp . (1153) | ORG | TODO |
| busang | Analysts and arbitrageurs immediately ruled out Barrick Gold Corp and Bre-X Minerals Ltd as Santa Fe saviors because they are locked in negotiations over their splitting Indonesia 's Busang vast gold deposit . (1161) | ORG | Bre-X , Barrick said to continue Busang talks . (2314) | LOC | TODO |
| jones | Then once we saw the Dow -LPR- Jones industrial average -RPR- was not about to crash , some buyers stepped in , " said a trader , referring to Federal Reserve Chairman Alan Greenspan , whose comments that assets were " irrationally exhuberant " upset financial markets worldwide . (1387) | MISC | CRICKET - JONES HITS CENTURY AS VICTORIA FIGHT BACK . (3172) | PER | TODO |
| santiago | SANTIAGO 1996-12-05 (1462) | PER | -- Roger Atwood , Santiago newsroom +56-2-699-5595 x211 (1471) | LOC | TODO |
| newcastle | COAL - Lantau Peak - 120,000 tones coal Hay Point or Newcastle / Kaohsiung 20-30/1 $ 5.25 and $ 5.85 fio respectively 40,000-35,000 / 28,000 shinc China Steel . (1908) | LOC | -LPR- Newcastle -RPR- (2790) | ORG | TODO |
| bangkok | BANGKOK 1996-11-29 (2262) | LOC | -- Bangkok newsroom -LPR- 662 -RPR- 652-0642 (2277) | MISC | TODO |
| wimbledon | Villa -RPR- , Les Ferdinand -LPR- Newcastle -RPR- , Efan Ekoku -LPR- Wimbledon -RPR- , (2792) | LOC | Wimbledon 16 9 4 3 29 17 31 (2949) | ORG | TODO |
| wallaby | The 34-year-old winger played his final game in a Wallaby jersey on Saturday but is currently a target for clubs eager to match London side Saracens who have already snapped up Francois Pienaar , Michael Lynagh and Philippe Sella . (2857) | ORG | RUGBY UNION - CAMPESE SIGNS OFF WITH TRY IN WALLABY ROMP . (3122) | LOC | TODO |
| oxford | Oxford 22 8 6 8 27 21 30 (2977) | LOC | Crystal Palace 2 Oxford 2 (3093) | ORG | TODO |
| charlton | Charlton 21 9 2 10 23 29 29 (2980) | ORG | SOCCER - ENGLISHMAN CHARLTON IS MADE AN HONORARY IRISHMAN . (3442) | PER | TODO |
| portsmouth | Portsmouth 22 7 5 10 25 29 26 (2986) | LOC | Sheffield United 1 Portsmouth 0 (3099) | ORG | TODO |

## Câu hỏi tự phân tích

- Cùng surface form xuất hiện trong ngữ cảnh nào khác nhau?
- Context nào giúp phân biệt các nhãn gold?
- Static representation sẽ thiếu thông tin gì trong các ví dụ này?
