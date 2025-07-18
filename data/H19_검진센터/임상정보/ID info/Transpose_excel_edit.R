
# library
library(openxlsx)
library(tidyverse)

# pre : import data set 
dir <- "file directory"
ID <- openxlsx::read.xlsx("C:\\Users\\user\\Desktop\\ID info\\20201109_우윤정_ID.xlsx")
diag <- openxlsx::read.xlsx("C:\\Users\\user\\Desktop\\ID info\\20201109_우윤정_정보.xlsx")

ID %>% as_tibble() # 확인 
diag %>% as_tibble() %>% 
  mutate(검진일 = as.Date(검진일, origin = "1899-12-30")) -> diag


diag %>% count(검진일) %>% as.data.frame()


ID %>% as_tibble() # 확인 
diag %>% as_tibble() %>% 
  mutate(검진일 = as.Date(검진일, origin = "1899-12-30")) -> diag


diag %>% count(검진일) %>% as.data.frame()


ID %>% 
  merge(diag, by = 'SEQ', all = TRUE) %>% as_tibble() %>%   select(-1) %>% 
  as.data.frame() -> transpose_data


ID_검진일 <- as.data.frame(matrix(99999, length(unique(transpose_data$ID)), 2))
names(ID_검진일) <- c("ID", "검진일")
data <- as.data.frame(matrix(99999, length(unique(transpose_data$ID)), length(unique(transpose_data$검사명))*2))
names(data) <- sort(c(unique(transpose_data$검사명), paste0(unique(transpose_data$검사명), "_진단")))

data <- as.data.frame(cbind(ID_검진일, data))


transpose_test <- function(data, index_column, return_data) { 
  
  date <- data[which(duplicated(data[, index_column]) != TRUE), "검진일"]
  index <- unique(data[, index_column])
  
  for (i in 1:length(index)) { 
    # https://stackoverflow.com/questions/27197617/filter-data-frame-by-character-column-name-in-dplyr
    data %>% 
      dplyr::filter(!!as.symbol(index_column) == index[i]) -> subset
    
    return_data[i, c(1,2)] <- c(index[i], as.character(date[i]))
    
    
    for ( j in 1:nrow(subset)) { 
      
      which_index <- which(subset$검사명[j] == colnames(return_data))
      return_data[i, which_index] <- subset$결과[j]
      
      if(which_index != 0) {
        return_data[i, (which_index + 1) ] <- subset$내용[j]
        
      }
    }
  }
  
  return(return_data)
  
}


test_result <- transpose_test(transpose_data, 'ID', return_data = data)


test_1 <- apply(test_result, MARGIN = 2, function(x) {gsub("\n$", "", x)})
test_1 %>% 
  write.csv("test_1.csv")









