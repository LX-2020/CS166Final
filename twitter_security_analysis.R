library(readr)

# reading in data sets
twitterData1 <- read_csv("~/Documents/SJSU/FA22/CS166/Project/twitterData1.csv")
twitterData2 <- read_csv("~/Documents/SJSU/FA22/CS166/Project/twitterData2.csv")
twitterData3 <- read_csv("~/Documents/SJSU/FA22/CS166/Project/twitterData3.csv")
twitterData4 <- read_csv("~/Documents/SJSU/FA22/CS166/Project/twitterData4.csv")
twitterData5 <- read_csv("~/Documents/SJSU/FA22/CS166/Project/twitterData5.csv")

# combining data into one dataframe
twitterData_combined <- rbind(twitterData1, twitterData2, twitterData3, twitterData4, twitterData5)

# read in sentiment analysis
twitterData_combined$sentiment_analysis <- read_csv("~/Documents/SJSU/FA22/CS166/Project/sentiment_analysis.csv")

# creating an empty column for fraud feature
twitterData_combined$isfraud <- c()

# factor variable assignments
twitterData_combined$user_advertiser_account_type <- as.factor(twitterData_combined$user_advertiser_account_type)
twitterData_combined$user_geo_enabled <- as.factor(twitterData_combined$user_geo_enabled)
twitterData_combined$user_has_custom_timelines <- as.factor(twitterData_combined$user_has_custom_timelines)
twitterData_combined$user/favourites_count <- as.factor(twitterData_combined$)
twitterData_combined$user/follow_request_sent <- as.factor(twitterData_combined$user/follow_request_sent)
twitterData_combined$user/followed_by <- as.factor(twitterData_combined$user/followed_by)
twitterData_combined$user/followers_count <- as.factor(twitterData_combined$user/followers_count)
twitterData_combined$user/following <- as.factor(twitterData_combined$user/following)
twitterData_combined$user/friends_count <- as.factor(twitterData_combined$user/friends_count)
twitterData_combined$user/geo_enabled <- as.factor(twitterData_combined$user/geo_enabled)
twitterData_combined$user/has_custom_timelines <- as.factor(twitterData_combined$user/has_custom_timelines)
twitterData_combined$user/id <- as.factor(twitterData_combined$user/id)
twitterData_combined$user/id_str <- as.factor(twitterData_combined$user/id_str)
twitterData_combined$user/is_translation_enabled <- as.factor(twitterData_combined$user/is_translation_enabled)
twitterData_combined$user/is_translator <- as.factor(twitterData_combined$user/is_translator)
twitterData_combined$user/lang <- as.factor(twitterData_combined$user/lang)
twitterData_combined$user/listed_count <- as.factor(twitterData_combined$user/listed_count)
twitterData_combined$user/protected <- as.factor(twitterData_combined$user/protected)
twitterData_combined$user/require_some_consent <- as.factor(twitterData_combined$user/require_some_consent)
twitterData_combined$user/media_count <- as.factor(twitterData_combined$user/media_count)
twitterData_combined$user/muting <- as.factor(twitterData_combined$user/muting)
twitterData_combined$user/name <- as.factor(twitterData_combined$user/name)
twitterData_combined$user/normal_followers_count <- as.factor(twitterData_combined$user/normal_followers_count)
twitterData_combined$user/notifications <- as.factor(twitterData_combined$user/notifications)

# sampling for the train test split
train_set <- sample(1:nrow(twitter_scrape_covid), 0.8*nrow(twitter_scrape_covid), replace = FALSE)

# split data into train and test split
train_split <- twitterData_combined[train_set,]
test_split <- twitterData_combined[-train_set,]

# train glm model using train split
glm_fit <- glm(
  is_fraud ~ full_text + 
    favorite_count + 
    hashtags_0 + 
    hashtags_1 + 
    hashtags_2 + 
    hashtags_3 + 
    hashtags_4 + 
    hashtags_5 + 
    hashtags_6 + 
    hashtags_7 + 
    hashtags_8 + 
    hashtags_9 + 
    reply_count + 
    retweet_count + 
    user_advertiser_account_type + 
    user_favourites_count + 
    user_followers_count + 
    user_friends_count + 
    user_geo_enabled + 
    user_has_custom_timelines + 
    user_listed_count + 
    user_location +
    user/profile_background_color +
    user/profile_background_image_url +
    user/profile_background_image_url_https +
    user/profile_background_tile +
    user/profile_banner_extensions_sensitive_media_warning +
    user/profile_banner_url +
    user/profile_image_extensions_sensitive_media_warning +
    user/profile_image_url +
    user/profile_image_url_https +
    user/profile_sidebar_border_color +
    user/profile_sidebar_fill_color +
    user/profile_text_color +
    user/profile_use_background_image +
    user/protected +
    user/require_some_consent +
    user/screen_name +
    favorite_count +
    reply_count +
    retweet_count +
    user/advertiser_account_type +
    user/ext_is_blue_verified +
    user/fast_followers_count +
    user/location +
    user/statuses_count, 
  family=binomial, 
  data=train_split
  )

# print a summary and save model
print(summary(glm_fit))
save(glm_fit, file="glm_fit.Rdata")

# run prediction on trained ML model
glm_probs <- predict(glm_fit, test_split, type="response")
