libname dataset "\\172.20.216.34\연구서버\연구검체\H13_구강_Oral";

proc import datafile="\\172.20.216.34\연구서버\연구검체\H13_구강_Oral\H13_구강암센터_Oral_대상자파일_20220728.xlsx" dbms=excel out=dataset.OC replace;Sheet="임상정보";run;

proc freq data= dataset.OC; 
          table NO	patno	NCCno	name	sex	age	HEIGHT	WEIGHT	BMI	birth	ABO	education	religion	Job	Doctor	KCD	OPdate	diagnosis	Dx	site	sitesite	shape	shapename
diagnosisdate	firtdianosisdate	CTxdate	treatment	RTxdate	G6	etctherapy	extraction	pathoStage	Tstage	Nstage	Nstagedetail	Nstagenode	stage	differentiation
Tumordepthmm	resectionmargin	ECS	boneinvasion	angioymphaticinvasion	perineuralinvasion	HPV	livetogether	married
smoking	smokingeaday	smokingperiod 	drinking	drinkingbottleevent	drinkingeventmonth	drinkingperiod 	father	mother	brothersister	children	delayed2ndprimary
concurrent2ndprimary	recurrence	extraction_2	recurrencedate	deathdate	extraction_3	remarks	etc	etcdisease	classification
; run;

