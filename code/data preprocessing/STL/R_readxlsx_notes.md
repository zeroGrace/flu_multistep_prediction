R语言中读文件时，最终文件名中不要出现“_”（下划线），可以有空格和字母。不然各种报错无法读取。
（用“openxlsx”读xlsx文件，以及不用包直接读csv文件均是如此；读xlsx文件时用"openxlsx"包读，不要用"xlsx"包，后者问题更多；读csv文件即使成功也容易乱码）
但写入好像不存在上述问题，写入的文件名中可以有下划线