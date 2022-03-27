library(openxlsx)
library(fpp2)
ILI_file = read.xlsx("C:/D/HUST/research_flu_forecast/influenza data/ILI_201001_202012/ILI for R.xlsx",sheet = 1)
weektag = ILI_file[,1]
n_ili = ILI_file$north_ILI
s_ili = ILI_file$south_ILI

#20% test 80% train
#cut_num = 426
#1/3 test 2/3 train
cut_num = 355

weektag_train = weektag[0:cut_num]
n_ili_train = n_ili[0:cut_num]
s_ili_train = s_ili[0:cut_num]

n_ili_ts = ts(n_ili_train, start = 2010, frequency = 52)
s_ili_ts = ts(s_ili_train, start = 2010, frequency = 52)

n_ili_stlobj = n_ili_ts %>% stl(t.window = 13, s.window = "periodic", robust = TRUE)
s_ili_stlobj = s_ili_ts %>% stl(t.window = 13, s.window = "periodic", robust = TRUE)

n_ili_stlobj %>% autoplot()
s_ili_stlobj %>% autoplot()

n_ili_sadj = seasadj(n_ili_stlobj) 
n_ili_s = seasonal(n_ili_stlobj)
n_ili_tc = trendcycle(n_ili_stlobj)
n_ili_r = remainder(n_ili_stlobj)
s_ili_sadj = seasadj(s_ili_stlobj) 
s_ili_s = seasonal(s_ili_stlobj)
s_ili_tc = trendcycle(s_ili_stlobj)
s_ili_r = remainder(s_ili_stlobj)

output = data.frame(weektag_train, n_ili_train, s_ili_train, n_ili_sadj, s_ili_sadj, n_ili_s, s_ili_s, n_ili_tc, s_ili_tc, n_ili_r, s_ili_r)
file = "C:\\D\\HUST\\research_flu_forecast\\data for coding\\STL results\\ILI_ori&stl_202012_train2test1_train.xlsx"
write.xlsx(output, file, sheetName = "sheet1", col.names = TRUE, row.names = FALSE)