# Upper Confidence Bound

# Importar DataSet

dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementar UCB
N = 10000
d = 10
number_of_selections = integer(d)
sums_of_rewards = integer(d)
ads_selected = integer(0)

total_rewards = 0

for(n in 1:N) {
  max_upper_bound = 0
  ad_selected = 0
  for (i in 1:d) {
    if (number_of_selections[i]>0) {
      average_reward = sums_of_rewards[i] / number_of_selections[i]
      delta_i = sqrt(3/2*log(n)/number_of_selections[i])
      upper_bound = average_reward + delta_i
    } else {
      upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound) {
      max_upper_bound = upper_bound
      ad_selected = i
    }
  }
  
  number_of_selections[ad_selected] =  number_of_selections[ad_selected] + 1
  ads_selected = append(ads_selected,ad_selected)
  reward = dataset[n,ad_selected]
  sums_of_rewards[ad_selected] =  sums_of_rewards[ad_selected] + reward
  total_rewards = total_rewards + reward
}

# Visualización resutlados
hist(ads_selected)
