libname dataset "\\172.20.216.34\연구서버\데이터입력시스템구축\Dummy data";

proc import datafile="\\172.20.216.34\연구서버\데이터입력시스템구축\Dummy data\설문지 dummy table.xlsx" dbms=excel out=dataset.L replace;Sheet="1.1 ver H13_for SAS";run;

proc freq data= dataset.L; 
          table version	new_pat_no	reg_no	visit_date	name	birth_date	age	sex	height	weight	LS_1	LS_3	LS_5	LS_6	LS_7	LS_8	LS_8_1	LS_8_2	LS_9	LS_9_1_1	LS_9_1_1a	LS_9_1_2	LS_9_1_2a
LS_9_1_3	LS_9_2_1	LS_9_2_2	LS_10	LS_10_1	LS_10_2	LS_11	LS_11_1	LS_12	LS_12_Fq1	LS_12_Fq2	LS_13	LS_13_1na	LS_13_1yr	LS_13_1Mo	LS_13_2na	LS_13_2yr	LS_13_2Mo	
LS_14	LS_14_1na	LS_14_1yr	LS_14_1Mo	LS_14_2na	LS_14_2yr	LS_14_2Mo	LS_15	LS_15Mu	LS_16	LS_16Mu	LS_17	LS_17Mu	LS_18	LS_18hy	LS_18hy_yr	LS_18hy_ag	LS_18hy_co	
LS_18di_sp	LS_18di_yr	LS_18di_ag	LS_18di_co	LS_18th_sp	LS_18th_yr	LS_18th_ag	LS_18th_co	LS_18ce_sp	LS_18ce_yr	LS_18ce_ag	LS_18ce_co	LS_18br_sp	LS_18br_yr	LS_18br_ag	
LS_18br_co	LS_18ca1	LS_18ca1_sp	LS_18ca1_yr	LS_18ca1_ag	LS_18ca1_co	LS_18ca2	LS_18ca2_sp	LS_18ca2_yr	LS_18ca2_ag	LS_18ca2_co	LS_18ch	LS_18ch_sp	LS_18ch_ag	
LS_18ch_co	LS_19	LS_19Fahy	LS_19Fahy_1	LS_19Fahy_2	LS_19Fahy_3	LS_19Fadi	LS_19Fadi_1	LS_19Fadi_2	LS_19Fadi_3	LS_19Fast	LS_19Fast_1	LS_19Fast_2	LS_19Fast_3	LS_19Fahe	
LS_19Fahe_1	LS_19Fahe_2	LS_19Fahe_3	LS_19Face	LS_19Face_1	LS_19Face_2	LS_19Face_3	LS_19Fath	LS_19Fath_1	LS_19Fath_2	LS_19Fath_3	LS_19Fabr	LS_19Fabr_1	
LS_19Fabr_2	LS_19Fabr_3	LS_19Faca	LS_19Faca_sp	LS_19Faca_1	LS_19Faca_2	LS_19Fach	LS_19Fach_sp	LS_19Fach_1	LS_19Fach_2	LS_19Fach_3	LS_20_	LS_20_1	LS_20_1_Fqwe	
LS_20_1_Fqda	LS_20_1_aM	LS_20_1_Mo	LS_20_2	LS_20_2_Fqwe	LS_20_2_Fqda	LS_20_2_aM	LS_20_2_Mo	LS_20_3	LS_20_3_Fqwe	LS_20_3_Fqda	LS_20_3_aM	LS_20_3_Mo	LS_20_4	
LS_20_4_Fqwe	LS_20_4_Fqda	LS_20_4_aM	LS_20_4_Mo	LS_20_5	LS_20_5_Fqwe	LS_20_5_Fqda	LS_20_5_aM	LS_20_5_Mo	LS_20_6	LS_20_6_Fqwe	LS_20_6_Fqda	LS_20_6_aM	LS_20_6_Mo	LS_20_7	
LS_20_7_sp	LS_20_7_Fqwe	LS_20_7_Fqda	LS_20_7_aM	LS_20_7_Mo	LS_20_8	LS_20_8_sp	LS_20_8_Fqwe	LS_20_8_Fqda	LS_20_8_aM	LS_20_8_Mo	LS_21sM2	LS_21sM2_yr	LS_21sM2_ea	
LS_21sM2_st	LS_21sM2_stea	LS_21sM2_styr	LS_21sM3	LS_21sM3_yr	LS_21sM3_ea	LS_21sM3_st	LS_21sM3_stea	LS_21sM3_styr	LS_21sM4	LS_21sM4_yr	LS_21sM4_ea	LS_21sM4_st	
LS_21sM4_stea	LS_21sM4_styr	LS_22	LS_22_hoMe	LS_22_hoMet	LS_22_work	LS_22_workt	LS_23	LS_23_yr	LS_23_Fq	LS_23_stopyr	LS_23_stopFq	LS_24_1	LS_24_1d	LS_24_1t	LS_24_1M
LS_24_2	LS_24_2d	LS_24_2t	LS_24_2M	LS_24_3	LS_24_3d	LS_24_3t	LS_24_3M	LS_24_4	LS_24_4d	LS_24_5	LS_24_5d	LS_24_6	LS_24_6d	LS_25_1	LS_25_2	LS_25_3	LS_25_4	LS_25_5	
LS_25_6	LS_25_7	LS_25_8	LS_25_9	LS_25_10	LS_25_11	LS_25_12	LS_25_13	LS_26_1	LS_26_1Mo	LS_26_2	LS_26_Fq	LS_27	LS_27_1na	LS_27_1Fq	LS_27_1ea	LS_27_2na	LS_27_2Fq	
LS_27_2ea	LS_27_3na	LS_27_3Fq	LS_27_3ea	LS_27_4na	LS_27_4Fq	LS_27_4ea	LS_27_5na	LS_27_5Fq	LS_27_5ea
; run;

