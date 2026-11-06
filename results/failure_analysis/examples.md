# Failure analysis — test split (post-hoc, descriptive)

C = RRF(BM25, ColPali) k=1 vs B = ColPali: 9 wins, 49 losses (nDCG@5).
C losses by tag: {'exact': 19, 'visual': 4, 'general': 26}; by mechanism: {'BM25 missed (gold > 5)': 42, 'BM25 also hit (top 5)': 7}.

## Favourable: queries where C beats ColPali (all)

- **What is the name of the laboratory?** — tag `general`; gold `hkbp0078`; gold rank A=3, B=14, C=4, C_w=14; query–gold OCR overlap 1.0.
  - gold-page OCR: “PRO DUC 1 PLEASE PRINT NAME/ADDRESS CORRECTIONS IN THIS SPACE. DOES THE CHANGE abbiy O PIN ION . YO ALL FAMILY MEMBERS (~) OR gust To You C) te ADDRESS CHANGED …”
  - BM25 top-1 `fyvb0228` OCR: “MEETING OF ADVISORY COMMISSION ON CONSUMER PROTECTION ELLINGTON AGRICULTURAL CENTER Conference Room - Office and Laboratory Building June 30, 1969 MORNING: 10:0…”

- **How many bullet points are there?** — tag `general`; gold `zpxl0226`; gold rank A=2, B=136, C=4, C_w=99; query–gold OCR overlap 0.33.
  - gold-page OCR: “DIRECTORS" MEETING Thursday, August 1st, 1929 AGENDA Minutes. Reports of Committees. a) Enforcement Committee b) Report to Department of Justice. Enforcement of…”
  - BM25 top-1 `fmbk0226` OCR: “. . REPRINTED FROM: MEDICAL WORLD NEWS APRIL 17, 1970 THEROSCLEROSIS The growing national debate over serum cholesterol, lipid metabolism, genetics, and life st…”

- **Which company’s memorandum is given?** — tag `general`; gold `jxjh0227`; gold rank A=4, B=6, C=5, C_w=6; query–gold OCR overlap 0.67.
  - gold-page OCR: “ae u 0 - INTER-OF FICE MEMORANDUM ye fw THE GREAT WESTERN SUGAR compan Hu \? ab ’ Your DATE: November 13 : All letters from geographic areas outlined below comi…”
  - BM25 top-1 `xgcc0228` OCR: “Campbell Soup Company File summary continued: On May ll, 1965, Dr. William J. Darby wrote another letter to Mr. W.B. Murphy, President of Campbell Soup, enclosi…”

- **What % of vests is available for conversion after 3 year?** — tag `exact`; gold `ynbx0223`; gold rank A=1, B=2, C=1, C_w=2; query–gold OCR overlap 0.6.
  - gold-page OCR: “ME ITC Limited Report on Corporate Governance Employee Stock Option Scheme The Company granted 59,69,437 Options during the financial year to the eligible emplo…”

- **what is the project number ?** — tag `general`; gold `rpgn0226`; gold rank A=1, B=2, C=1, C_w=2; query–gold OCR overlap 1.0.
  - gold-page OCR: “ORTHOPAEDICS RESEARCH REPORT COVER SHEET Metal on Metai Acetabular Cup Taper Fit Testing: Phase I Identity of Test Item: MOM Taper Samples 5° Shells (Qty 5): SD…”

- **What is the time of the reception?** — tag `general`; gold `thwm0227`; gold rank A=1, B=2, C=1, C_w=2; query–gold OCR overlap 0.5.
  - gold-page OCR: “You are cordially invited to attend a reception honoring G. DONALD WHEDON, M.D. on the occasion of his resignation as Director National Institute of Arthritis, …”

- **Who is from Indiana?** — tag `exact`; gold `ggbm0227`; gold rank A=1, B=2, C=1, C_w=1; query–gold OCR overlap 1.0.
  - gold-page OCR: “Thomas W. Christopher, Esq. Dean University of Alabama Law School Tuscaloosa, Alabama 35486 Tel. #: 205-348-6010 John.} ...Coburh.s..T.D.;,.D<D. - St. James Epi…”

- **When is the remittance due ?** — tag `general`; gold `qtyp0227`; gold rank A=1, B=2, C=1, C_w=2; query–gold OCR overlap 1.0.
  - gold-page OCR: “DIVISION OF BEMO SHIPPING COMPANY, INC. SUITE 1767 » ONE WORLD TRADE CENTER « NEW YORK, N.Y. 10048 oo os W. R. KEATING & COMPANY SHIPMENTS HANDLED TO AND TEL: (…”

- **What is the name of the factory ?** — tag `general` (duplicated text); gold `lgph0227`; gold rank A=2, B=5, C=3, C_w=5; query–gold OCR overlap 0.5.
  - gold-page OCR: “FORM 443 - 8-57 - PADS 100 WATER ANALYSIS 3............Factory Kind of Water: River Water Description of Sauipe: Average sample for 1961-62 campaign Source of W…”
  - BM25 top-1 `mfmh0227` OCR: “PENNANT RATING PREMIUMS AND PENALTIES - (PENALTIES UNDERLINED FIRST PERIOD ENDING OCTOBER 7, 1963 Pennant Rating Mol. Comp. Period & True Sacch. Factory Dryer L…”


## Unfavourable: the 10 largest losses of C vs ColPali

- **What is No. 2-26 of subheading H ?** — tag `exact`; gold `kkwg0227`; gold rank A=342, B=1, C=3, C_w=1; query–gold OCR overlap 0.0.
  - gold-page OCR: “RESEARCH IN ANALYTICAL CHEMISTRY: No. 2-21: Analytical chemistry methodology and its application to medicaly and nutritional research No. 2- Development of new …”
  - BM25 top-1 `kkpj0226` OCR: “Additional copies available at A Proposal for h br iting to: . . no charge By writing fo Restructuring Children’s The Quaker Oats Company Television P.O. Box 34…”

- **What Tobacco brand of GPI is shown in the picture?** — tag `visual`; gold `znbx0223`; gold rank A=31, B=1, C=3, C_w=1; query–gold OCR overlap 0.2.
  - gold-page OCR: “Inspired individuals share their story- what drove them to succeed against the odds and how they realized their goals. “The launch of three new brands has been …”
  - BM25 top-1 `xhxj0037` OCR: “PHILIP MORRIS U.S. A. INTER-OFFICE CORRESPONDENCE Richmond, Virginia To: Mr. R. P. Heretick Date: June 5, 1990 From: W. A. Geiszler cg90. 02997 Subject: Filter …”

- **What is the VO. NO.?** — tag `exact`; gold `rzgm0227`; gold rank A=137, B=1, C=3, C_w=1; query–gold OCR overlap 0.0.
  - gold-page OCR: “Citizens’ Commission ON Science, Ae Ol, & Law anv THE Food Supp “heck & YO. NO.73-170 _ December —20,_______1973 Kindly have check drawn to order of AVIS for$ 2…”
  - BM25 top-1 `ghlw0228` OCR: “Gsons To BE SENT By tea,vo THE BRONZE GALLERY ~ DEALERS IN NIGERIA ARTS DQ, 9.4 DARTY- 39, CAMPBELL STREET, P. 0. BOX 3013 LAGOS—NIGERIA. TEL. 22871 CASH’ SALES…”

- **What is the announcement on?** — tag `general`; gold `kkbm0227`; gold rank A=336, B=3, C=6, C_w=4; query–gold OCR overlap 0.0.
  - gold-page OCR: “/ Now you can bring your students up to date with the f newest development in nutrition and genetics... INBORN ERRORS OF METABOLISM a by Charles R. Scriver, M.D…”
  - BM25 top-1 `tllk0226` OCR: “TO: ISRF Members DATE: October 8, 1976 SUBJECT: Recent Literature on Diabetes DIABETES ETIOLOGY * Please add to your loose-leaf folder the following: - "Nutriti…”

- **For which brand  of soft drink is the product information provided?** — tag `general`; gold `hjpj0226`; gold rank A=5, B=1, C=3, C_w=1; query–gold OCR overlap 0.5.
  - gold-page OCR: “INGREDIENTS DATA Pepsi-Cola Product information Ingredients are given per fluid ounce of bottled or canned beverage Diet fMlountain Component’! Pepsi-Cola Diet …”
  - BM25 top-1 `yhpj0226` OCR: “MRI dL SEGLLMAN DIRECTOR/L OS AN DRAFT: DO NOT USE PUBLICLY UNTIL ALL FACTS ARE CONFIRMED AND DOCUMENTED A STERN CONCERN FACT MEMO ON SOFT DRINKS The following …”

- **What is the cost of postage ( see the postal stamp at top left)?** — tag `visual`; gold `phjc0228`; gold rank A=252, B=3, C=7, C_w=4; query–gold OCR overlap 0.0.
  - gold-page OCR: “(empty OCR)”
  - BM25 top-1 `jzfv0228` OCR: “BUSINESS REPLY MAIL FIRST CLASS PERMIT NO. 3275 HOUSTON, TEXAS 77030 POSTAGE WILL BE PAID BY ADDRESSEE The University of Texas System Cancer Center M. D. Anders…”

- **What is the main heading of the document given under APPENDIX V-9?** — tag `exact`; gold `snyc0227`; gold rank A=27, B=1, C=3, C_w=1; query–gold OCR overlap 0.2.
  - gold-page OCR: “Malayan Chermai Chabai; Chili Lada Bawang Sagu Gandum Gelugur Janggus Jarak Jelai Jerok Jintan Buah sireh kaya Kayu manis Pokok nyiur; Pokok kelapa Petola hutan…”
  - BM25 top-1 `slcn0226` OCR: “Scientific Review of the Proposed Risk Assessment Bulletin from the Office of Management and Budget http://www.nap.edu/catalog/1 1811.html Impacts, 55 Reference…”

- **Which year was the meeting held?** — tag `general`; gold `jhhd0227`; gold rank A=57, B=4, C=8, C_w=4; query–gold OCR overlap 0.33.
  - gold-page OCR: “EXECUTIVE FACULTY Annual Meeting 1975 Page 10 Emeritus Titles - Continued Dr. Edith C. Robinson, Assistant Professor Emeritus of Clinical Pediatrics. Dr. Harold…”
  - BM25 top-1 `qznc0227` OCR: “NATIONAL INSTITUTES OF HEALTH Bethesda 14, Maryland May 1, 1959 MEMORANDUM To : Dr. Youmans Dr. Kampmeier Dr. Darby Dr. Shideman Dr. McGanity Dr. le Brocquy Dr.…”

- **When was the meeting held?** — tag `general`; gold `sncj0037`; gold rank A=21, B=4, C=7, C_w=4; query–gold OCR overlap 0.5.
  - gold-page OCR: “% BROWN & WILLIAMSON TOBACCO CORPORATION RESEARCH & DEVELOPMENT MEETING REPORT SUBJECT : Reduced Ignition Propensity/498 PRESENT: J. Wu, L. C. Chao, S. W. Brook…”
  - BM25 top-1 `qznc0227` OCR: “NATIONAL INSTITUTES OF HEALTH Bethesda 14, Maryland May 1, 1959 MEMORANDUM To : Dr. Youmans Dr. Kampmeier Dr. Darby Dr. Shideman Dr. McGanity Dr. le Brocquy Dr.…”

- **What is the amount of check?** — tag `general`; gold `zggb0228`; gold rank A=6, B=4, C=6, C_w=3; query–gold OCR overlap 0.5.
  - gold-page OCR: “Jee NELST Pe LON, FOUNDATION FING: NINETY-NINE PARK AVENUE - NEW YORK,N.Y. - 10016 - 212 --687-4830 January 5, 1971 Dr. Alexander Heard, Chancellor Vanderbilt U…”
  - BM25 top-1 `tllg0227` OCR: “Seyeg ed ee pie 7 GAO oss we PUBLIC VOUCHER FOR PURCHASES AND mime eae NO sH 200-05 SERVICES OTHER THAN PERSONAL yoy : Use continuation sheet(s) if necessary BU…”


## Every query where C_w (weighted, w=0.95, k=20) differs from ColPali

- **What is the amount of check?** — tag `general`; gold `zggb0228`; gold rank A=6, B=4, C=6, C_w=3; query–gold OCR overlap 0.5.
  - gold-page OCR: “Jee NELST Pe LON, FOUNDATION FING: NINETY-NINE PARK AVENUE - NEW YORK,N.Y. - 10016 - 212 --687-4830 January 5, 1971 Dr. Alexander Heard, Chancellor Vanderbilt U…”
  - BM25 top-1 `tllg0227` OCR: “Seyeg ed ee pie 7 GAO oss we PUBLIC VOUCHER FOR PURCHASES AND mime eae NO sH 200-05 SERVICES OTHER THAN PERSONAL yoy : Use continuation sheet(s) if necessary BU…”

- **What is the announcement on?** — tag `general`; gold `kkbm0227`; gold rank A=336, B=3, C=6, C_w=4; query–gold OCR overlap 0.0.
  - gold-page OCR: “/ Now you can bring your students up to date with the f newest development in nutrition and genetics... INBORN ERRORS OF METABOLISM a by Charles R. Scriver, M.D…”
  - BM25 top-1 `tllk0226` OCR: “TO: ISRF Members DATE: October 8, 1976 SUBJECT: Recent Literature on Diabetes DIABETES ETIOLOGY * Please add to your loose-leaf folder the following: - "Nutriti…”

- **What is the cost of postage ( see the postal stamp at top left)?** — tag `visual`; gold `phjc0228`; gold rank A=252, B=3, C=7, C_w=4; query–gold OCR overlap 0.0.
  - gold-page OCR: “(empty OCR)”
  - BM25 top-1 `jzfv0228` OCR: “BUSINESS REPLY MAIL FIRST CLASS PERMIT NO. 3275 HOUSTON, TEXAS 77030 POSTAGE WILL BE PAID BY ADDRESSEE The University of Texas System Cancer Center M. D. Anders…”

- **Who is from Indiana?** — tag `exact`; gold `ggbm0227`; gold rank A=1, B=2, C=1, C_w=1; query–gold OCR overlap 1.0.
  - gold-page OCR: “Thomas W. Christopher, Esq. Dean University of Alabama Law School Tuscaloosa, Alabama 35486 Tel. #: 205-348-6010 John.} ...Coburh.s..T.D.;,.D<D. - St. James Epi…”

- **Who is the holder of this curriculum vitae?** — tag `general`; gold `prxn0226`; gold rank A=199, B=2, C=3, C_w=3; query–gold OCR overlap 0.0.
  - gold-page OCR: “{O87 7900-109] Honors and Awards John 3 the Hip Society: nan ort a ening of C Hip Award for Orthopaedic ine Problem in To fing Fellow of the American Orthopaedi…”
  - BM25 top-1 `mlcj0227` OCR: “y) Education: colleges, universities (give degrees and dates), special training University of North Carolina, (1956- “39; AB), Duke Medical School, 59> 63, Medi…”

