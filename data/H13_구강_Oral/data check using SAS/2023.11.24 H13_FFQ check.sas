libname dataset "\\172.20.216.34\연구서버\연구검체\H13_구강_Oral\SAS 임상정보 확인";

proc import datafile="\\172.20.216.34\연구서버\연구검체\H13_구강_Oral\SAS 임상정보 확인\FFQ_항목별.xlsx" dbms=excel out=dataset.ffq replace;Sheet="FFQ_amount";run;

proc freq data= dataset.ffq; 
          table qa1	qa2	qa3	qa4	qa5	qa6	qa7	qa8	qa9	qa10	qa11	qa12	qa13	qa14	qa15	qa16	qa17	qa18	qa19	qa20	qa21	qa22	qa23	qa24	qa25	qa26	qa27	qa28	qa29	qa30	qa31	qa32	qa33	qa34	qa35	qa36	qa37	qa38	qa39	qa40	qa41	qa42	qa43	qa44	qa45	qa46	qa47	qa48	qa49	qa50	qa51	qa52	qa53	qa54	qa55	qa56	qa57	qa58	qa59	qa60	qa61	qa62	qa63	qa64	qa65	qa66	qa67	qa68	qa69	qa70	qa71	qa72	qa73	qa74	qa75	qa76	qa77	qa78	qa79	qa80	qa81	qa82	qa83	qa84	qa85	qa86	qa87	qa88	qa89	qa90	qa91	qa92	qa93	qa94	qa95	
; run;

