require(tidyverse)
require(ggpubr)
require(pracma)

df.rans = read.csv("../data/preprocessed_data/rans_raw.csv") %>%
  rename(x='Points.0', y='Points.1', k='kMean') %>%
  select(x, y, k)

df.dns = read.csv("../data/preprocessed_data/dns_raw.csv") %>%
  rename(x='x_c', y='y_c', k='k_u2') %>%
  select(x, y, k)

df.rans %>% head
df.dns %>% head

df.rans.normalized = df.rans %>% mutate(x = round(x / .2,2), y = round(y / .2, 5))
df.dns.normalized = df.dns %>% mutate(x = x / 1, y = y / 1)

df.all = rbind(
  df.rans %>% mutate(source='rans'),
  df.dns %>% mutate(source='dns')
)

df.all.normalized = rbind(
  df.rans.normalized %>% mutate(source='rans'),
  df.dns.normalized %>% mutate(source='dns')
)

ggplot(df.all, aes(x=y)) + geom_histogram() + facet_wrap(x ~ source)

ggplot(df.all.normalized, aes(x=y)) + geom_histogram() + facet_wrap(x ~ source)


ggplot(df.all.normalized, aes(x=y, y=k, color=source)) + geom_line() + facet_wrap(x ~ .) + scale_x_continuous(trans='log10') + scale_y_continuous(trans='log10')

ggplot(df.dns.normalized, aes(x=y, y=k)) + geom_line() + scale_x_continuous(trans='log10') + scale_y_continuous(trans='log10') + facet_wrap(x ~ .)


ggplot(df.all.normalized, aes(x=y, y=k, color=source)) + geom_line() + facet_wrap(x ~ .) + xlim(0.05, 0.1) + scale_y_continuous(trans='log10')

ggplot(df.all.normalized, aes(x=y, y=k, color=source)) + geom_line() + facet_wrap(x ~ .) + xlim(0.05, 0.3) +  scale_y_continuous(trans='log10')

ggplot(df.rans.normalized %>% filter(y < 0), aes(x=y, y=k)) + geom_line() + facet_wrap(x ~ .) + scale_y_continuous(trans='log10')


# df.dns.ranged = df.dns.normalized %>% filter(y >= 0.05 & y < 0.3)
df.dns.ranged = df.dns.normalized %>% filter(y >= 0.05 & y < 0.1) %>% arrange(x, y)
df.rans.ranged = df.rans.normalized %>% filter(y >= 0.05 & y < 0.1) %>% arrange(x, y)


ggplot(df.all.normalized, aes(x=y, y=k, color=source)) + geom_line() + facet_wrap(x ~ .) + xlim(0.05, 0.1)

write_csv(df.dns.ranged, "../data/normalized_dns_in_range.csv")
write_csv(df.rans.ranged, "../data/normalized_rans_in_range.csv")

my.polyfit = function(df, n) {
  poly.coeff = polyfit(df$y, df$k, n)
  poly.func = Vectorize(function(x) { sum(x^(n:0) * poly.coeff) })
  poly.func(df$y)
}

(
df.dns.polys = df.dns.ranged %>% 
  group_by(x) %>%
  mutate(poly3=my.polyfit(cur_data(), 3)) %>%
  mutate(poly5=my.polyfit(cur_data(), 5)) %>%
  mutate(poly7=my.polyfit(cur_data(), 7)) %>%
  ungroup()
) 

(
ggplot(
  df.dns.polys %>% pivot_longer(c(-x,-y), names_to="k.source", values_to="k"),
  aes(x=y, y=k, color=k.source)
  ) + 
  geom_line() + 
  scale_y_continuous(trans='log10') +
  facet_wrap(x ~ .)
)