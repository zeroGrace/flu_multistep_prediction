library(PMCMR)
library(readxl)
library(openxlsx)

series_c = c("Nori", "Sori")
metric_c = c("MAPE","RMSE","PWD","Outbreak_MAE")
row_n = c("H2","H3","H4","H5","H6","H7","H8","H9","H10")
col_n = c("Total","SVR","MLP")

s_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Statistical test/"

for (series in series_c){
  wb_F = createWorkbook()
  
  for (metric in metric_c) {
    

    #choose open file path
    if(metric %in% c("MAPE","RMSE")){
      o_path = "C:/D/HUST/research_flu_forecast/experiment/result/New/Statistical Metrics/total_metric_gather/"
    } else {
      o_path = paste0("C:/D/HUST/research_flu_forecast/experiment/result/New/",metric,"/total_metric_gather/")
    }

    wb_N_total = createWorkbook()
    wb_N_SVR = createWorkbook()
    wb_N_MLP = createWorkbook()

    rF_step = c()
    for (h in 2:10) {
      sheet_name = paste0("H",as.character(h))   #paste默认空格连接，paste0默认无缝连接
      
      f_name = paste(series,"total",metric,"gather.xlsx",sep = "_")
      
      re =  read_excel(paste0(o_path,f_name),sheet = sheet_name, range = "B1:G21")
      re_m = as.matrix(re)
      re_m_SVR = re_m[,1:3]
      re_m_MLP = re_m[,4:6]

      rF_total = friedman.test(re_m)
      rF_SVR = friedman.test(re_m_SVR)
      rF_MLP = friedman.test(re_m_MLP)
      rN_total = posthoc.friedman.nemenyi.test(re_m)
      rN_SVR = posthoc.friedman.nemenyi.test(re_m_SVR)
      rN_MLP = posthoc.friedman.nemenyi.test(re_m_MLP)
      
      #print(paste(f_name,sheet_name))
      #print(rF_total)
      #print(rN)
      
      addWorksheet(wb_N_total, sheetName = sheet_name)
      writeData(wb_N_total, sheet = sheet_name, rN_total$p.value, colNames = TRUE, rowNames = TRUE)
      addWorksheet(wb_N_SVR, sheetName = sheet_name)
      writeData(wb_N_SVR, sheet = sheet_name, rN_SVR$p.value, colNames = TRUE, rowNames = TRUE)
      addWorksheet(wb_N_MLP, sheetName = sheet_name)
      writeData(wb_N_MLP, sheet = sheet_name, rN_MLP$p.value, colNames = TRUE, rowNames = TRUE)

      rF_step = c(rF_step, c(rF_total$p.value, rF_SVR$p.value, rF_MLP$p.value))  #始终是一维向量
    }
    
    sf_N_total_name = paste(series,"total",metric,"stat_Nemenyi.xlsx",sep = "_")
    file_N_total = paste0(s_path,sf_N_total_name)
    saveWorkbook(wb_N_total, file_N_total)
    
    sf_N_SVR_name = paste(series,"SVR",metric,"stat_Nemenyi.xlsx",sep = "_")
    file_N_SVR = paste0(s_path,sf_N_SVR_name)
    saveWorkbook(wb_N_SVR, file_N_SVR)

    sf_N_MLP_name = paste(series,"MLP",metric,"stat_Nemenyi.xlsx",sep = "_")
    file_N_MLP = paste0(s_path,sf_N_MLP_name)
    saveWorkbook(wb_N_MLP, file_N_MLP)

    rF_m = matrix(rF_step, nrow = 9, byrow = TRUE, dimnames = list(row_n, col_n))  #将一维向量叠成矩阵
    
    addWorksheet(wb_F,sheetName = metric)
    writeData(wb_F,sheet = metric, rF_m, colNames = TRUE, rowNames = TRUE)
  }
  
  sf_F_name = paste(series,"total","stat_Friedman.xlsx",sep = "_")
  file_F = paste0(s_path,sf_F_name)
  saveWorkbook(wb_F, file_F)
}
