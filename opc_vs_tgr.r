library(ggplot2)
library(gridExtra)
library(viridis)
library(sf)
library(cowplot)
library(stringi)

# this comes from some interesting relationships.
#
# In [55]: plt.scatter(df_floats["total_grape_revenue"].apply(lambda x: math.sqrt(math.sqrt(x))), df_floats["total_operating_costs"])
#     ...: plt.show()

# In [56]: plt.scatter(df_floats["total_grape_revenue"].apply(lambda x: math.sqrt(math.sqrt(x))), df_floats["total_operating_costs"].apply(lambda x: math.sqrt(math.sqrt(x))))
#     ...: plt.show()

# In [57]: plt.scatter(df_floats["total_grape_revenue"].apply(lambda x: math.sqrt(math.sqrt(x))), df_floats["water_used"].apply(lambda x: np.log(x+1)))
#     ...: plt.show()

df <- read.csv("dfb.csv")
df <- df[df$total_operating_costs != 0, ]
df <- df[df$total_grape_revenue != 0, ]

total <- ggplot(
    df
    , aes(
        x = total_operating_costs^(1 / 3)
        , y = total_grape_revenue^(1 / 3)
        , col = giregion)
        ) +
        xlim(-2.5, 2.5) +
        ylim(-2.5, 2.5) +
        geom_point(size = .5) +
        xlab("Total Operating Costs") +
        ylab("Total Grape Revenue") +
        ggtitle("Operating cost vs Revenue") +
        theme_light() +
        theme(panel.grid.minor = element_blank()
            , legend.title = element_blank()
            , legend.position = "top", legend.direction = "horizontal")

by_area <- ggplot(
    df
    , aes(
        x = total_operating_costs^(1 / 3)
        , y = total_grape_revenue^(1 / 3)
        , col = giregion)
        ) +
        xlim(-2.5, 2.5) +
        ylim(-2.5, 2.5) +
        geom_point(size = .5) +
        xlab("Total Operating Costs") +
        ylab("Total Grape Revenue") +
        ggtitle("Operating cost vs Revenue by area") +
        theme_light() +
        theme(panel.grid.minor = element_blank()
            , legend.title = element_blank()
            , legend.position = "top", legend.direction = "horizontal")

pdf("yield_verse_value.pdf")
 grid.arrange(total, by_area, ncol=2)
dev.off()