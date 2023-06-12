print("Starting script")
import time
time.sleep(1)
import pandas as pd
import numpy as np

path = "data"
indicadores = pd.read_csv(f"{path}/indicadores.csv")
print("loaded HIV data")

common_cols = {
    "Código": "code", 
    "Nome Município": "name",
}

cols_hiv_mf = {
    "Casos":"hiv_total",
    "Casos M":"hiv_m",
    "Casos F":"hiv_f"#,
    #"Tx Det":"hiv_dr_total",
    #"Tx Det M":"hiv_drm",
    #"Tx Det F":"hiv_drf"
}

cases = pd.DataFrame()
# city code and name
for col_port, col_eng in common_cols.items():
    cases.loc[:,col_eng] = indicadores[col_port]

years_hiv = range(2013,2023)
for year in years_hiv:
    cases[year] = 0
# create panel
cases = (cases.melt(id_vars=["code","name"],
                        value_vars=years_hiv)
          .rename(columns={"variable":"year"})
          .drop("value",axis=1))

sexuality_codes = {
    "Casos Categ. Exp. Hierarq. 10":"hiv_h"
}

## add columns in 2nd file
def add_rows(final_df, original_df, dict_translations, years, complete=[]):
    for code, group in dict_translations.items():
        #print(code, group)

        cases_1group_all_years = pd.Series()
        for year in years:
            col_original = f"{code} {year}"
            # the column year will have the cases of that year for that group
            # so that I can just melt bellow
            group_year_i = original_df[col_original]
            cases_1group_all_years = pd.concat([cases_1group_all_years, group_year_i],
                                               axis=0)

        for y in complete:
            cases_1group_all_years = pd.concat([cases_1group_all_years,
                                               pd.Series([np.nan]*5657)],
                                               axis=0)
            

        len_cases = len(cases_1group_all_years)
        final_df = (final_df.assign(
                new_col=cases_1group_all_years.values)
                .rename(columns={"new_col":group}))
    return final_df

indicadores_2 = pd.read_csv(f"{path}/indicadores_cont_2.csv")

cases = add_rows(cases, indicadores, cols_hiv_mf, years_hiv)
cases = add_rows(cases, indicadores_2, sexuality_codes, years_hiv)



print("loading Hepatites data")
hepatite_xls = pd.ExcelFile(f"{path}/hepatite.xlsx")

hep_dados = pd.read_excel(hepatite_xls, "DADOS")
hep_dados_cont = pd.read_excel(hepatite_xls, "DADOS CONTINUAÇÃO 1")
#hep_popm = pd.read_excel(hepatite_xls, "Pop.M")
#hep_popf = pd.read_excel(hepatite_xls, "Pop.F")

hep_AB_cols = {
    "Hepatite A":"hep_a_total",
    "Hepatite B":"hep_b_total",
    "Hep B M":"hep_b_m",
    "Hep B F":"hep_b_f",
    "Tx Det Hep A":"hep_a_dr",
    "Tx Det Hep B":"hep_b_dr",
    "Tx Det M Hep A M":"hep_a_dr_m",
    "Tx Det M Hep B M":"hep_b_dr_m",
    "Tx Det F Hep A F":"hep_a_dr_f",
    "Tx Det F Hep B F":"hep_b_dr_f"
}
hep_C_cols = {
    "Hepatite C":"hep_c_total",
    "Tx Det Hep C":"hep_c_dr",
    "Tx Det M Hep C M":"hep_c_dr_m",
    "Tx Det F Hep C F":"hep_c_dr_f"
}

hep_populacao = pd.read_excel(hepatite_xls, "População")
population = hep_populacao[["Código","Nome Município",2016]]
population = population.rename(columns={"Código":"code",
                                "Nome Município":"name",
                                2016:"population"})

years_hep = range(2013, 2021)#13-20

cases = add_rows(cases, hep_dados, hep_AB_cols, years_hep, complete=[2021,2022])
cases = add_rows(cases, hep_dados_cont, hep_C_cols, years_hep, complete=[2021,2022])


# TREATMENT VARIABLE / CONTROLS
cases = cases.set_index("code")
cases.loc[:, "region_type"] = "city"
cases.loc[cases.index < 100, "region_type"] = "state"
cases.loc[cases.index <= 10, "region_type"] = "region"
cases.loc[55, "region_type"] = "brazil"
cases = cases.reset_index()

# add percent treat

import pandas as pd
prep = pd.DataFrame()
for year in range(2018, 2023):
    df = pd.read_csv(f"data/prep{year}.csv")
    df["year"] = year
    prep = pd.concat([prep, df])
prep_data = prep.groupby(["city_code", "year"]).tail(1)

new_cases = (prep_data.set_index(["city_code","year"])
  .sort_values(["city_code","year"])
    .groupby("city_code")
    .agg( {"following":"diff"} )
    .reset_index()
    .rename(columns={"following":"new_users"}))
prep_data = prep_data.merge(new_cases, left_on=["city_code","year"],
                            right_on=["city_code","year"],how="left")

prep_data.loc[prep_data["year"] == 2018, "new_users"] = prep_data["following"]

cases = cases.merge(prep_data, left_on=["code","year"],
                right_on=["city_code","year"],
                how="left")
cases.loc[cases["hiv_h"] == 0, "pct_new_users"] = 0
cases.loc[cases["hiv_h"] != 0, "pct_new_users"] = cases["new_users"]/cases["hiv_h"]
cases.loc[cases["new_users"] < 0, "pct_new_users"] = 0

cases.loc[cases["hiv_h"] == 0, "pct_following"] = 0
cases.loc[cases["hiv_h"] != 0, "pct_following"] = cases["following"]/cases["hiv_h"]

cases = cases.query("code != 0")

cases = cases.merge(population,
            left_on=["code","name"],
            right_on=["code","name"],
            how="left")

## CONTROLS

print("loading controls")
cities = cases.query("region_type == 'city'")
cities["state_code"] = cities["code"].astype(str).str.slice(0,2)

## ieps = institute for health policy studies
ieps = pd.read_csv("data/ieps.csv")
rename_ieps = {
    "codmun":"code",
    "ano":"year",
    "desp_tot_saude_pc_mun":"health_expend",
    "gasto_pbf_pc_def":"cash_transfer_expend",
    "pib_cte_pc":"gdpp",
    "pct_pop20a24":"pop20_24"
}
ieps = ieps.rename(columns=rename_ieps)
ieps = ieps.replace(' ',pd.NA)
ieps = ieps[["code","year","health_expend",
             "cash_transfer_expend","gdpp","pop20_24"]]

numeric_cols = ["health_expend","cash_transfer_expend",
                "gdpp","pop20_24"]
for c in numeric_cols:
    ieps.loc[ieps[c].notna(),c] = ieps.loc[ieps[c].notna(),c].str.replace(",",".").astype(float).round(2)

cities = cities.merge(ieps, on=["code","year"], how="left")

## COVID DATA
covid = pd.read_csv("data/covid.csv")
covid = covid.query("place_type == 'city'")
covid["state_code"] = covid["city_ibge_code"].astype(str).str.slice(0,2)
covid["date"] = pd.to_datetime(covid["date"])
covid = covid.rename(columns={"date":"year"})
covid = covid.sort_values(["state","city","year"])
covid = covid.set_index(["state","city"])
covid_cases = (covid.groupby(["state_code", "city",covid["year"].dt.year])
                 .agg({"confirmed":"last"}).reset_index())
covid_cases = covid_cases.rename(columns={"confirmed":"covid_cases"})
cities = cities.merge(covid_cases, left_on=["state_code","name","year"],
                          right_on=["state_code","city","year"], how="left")
cities.loc[cities["covid_cases"].isnull(), "covid_cases"] = 0

def set_covid_22(df):
    """for cities without covid data for 22, set 21 value + 20%"""
    df = df.set_index("year")
    if df.loc[2022, "covid_cases"] == 0:
        cases_21 = df.loc[2021, "covid_cases"]
        df.loc[2022, "covid_cases"] = 1.2*cases_21
    return df

cities = cities.groupby("code").apply(set_covid_22).drop("code",axis=1).reset_index()
cities["covid_per_capita"] = cities["covid_cases"]/cities["population"]


## MERGE WITH REGION INFO

ibge = pd.read_excel("data/regioes_ibge.xls")
ibge["state"] = ibge["CD_GEOCODI"].astype("str").str.slice(0, 2)

cities.loc[:,"state"] = cities["code"].astype("str").str.slice(0, 2)

ibge = ibge[["nome_mun", "state", "nome_rgi", "cod_rgi"]]
ibge = ibge.rename(columns={"nome_mun":"name",
                            "nome_rgi":"region_name",
                            "cod_rgi":"region_code"})
cities = cities.merge(ibge, on=["state","name"],
                        how="inner")
cities.to_csv("data/cities.csv", mode="w", index=False)

## REGIONS

#regions
import numpy as np
regions = (cities.set_index(["region_code","region_name","year"])
        .groupby(["region_code","region_name","year"])
        .agg({"hiv_total":"sum",
              "hiv_h":"sum",
              "hiv_f":"sum",
              "hiv_m":"sum",
              "hep_b_total":"sum",
              "hep_c_total":"sum",
              "hep_b_m":"sum",
              "hep_b_f":"sum",
              "following":"sum",
              "loss_of_follow":"sum",
              "new_users":"sum",
              "population":"sum"
             })
        .reset_index())

regions.loc[regions["hiv_h"] == 0, "pct_new_users"] = 0
regions.loc[regions["hiv_h"] != 0, "pct_new_users"] = regions["new_users"]/regions["hiv_h"]
regions.loc[regions["new_users"] < 0, "pct_new_users"] = 0

regions.loc[regions["hiv_h"] == 0, "pct_following"] = 0
regions.loc[regions["hiv_h"] != 0, "pct_following"] = regions["following"]/regions["hiv_h"]

regions = regions.set_index("region_code")

def average_by_region(var):
    return (cities.query(f"{var}.notna()").groupby(["region_code","year"])
            .apply(lambda x: np.average(x[var], weights=x.population))
    .reset_index().rename(columns={0:var} ))

control_vars = ["health_expend","cash_transfer_expend","gdpp","pop20_24","covid_per_capita"]

for var in control_vars:
    ## compute the average of the outcome by region
    ## ieps has no data for 2022
    var_df = average_by_region(var)
    regions = regions.merge(var_df, on=["region_code","year"],how="left")
    
regions["state_code"] = regions["region_code"].astype(str).str.slice(0,2)
regions.to_csv("data/regions.csv", mode="w", index=False)


## PLOT CONFIGS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

plt.rcParams["figure.figsize"] = [4,3]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
pd.set_option('display.max_columns', 500)
sns.set_style("whitegrid")
plt.style.use('./graph_config.mplstyle')

HET_COLOR = "#00A4CCFF"# "#00A4CCFF"
GAY_COLOR = "#F95700FF" # "#F95700FF"
palette_hetgay = [HET_COLOR,GAY_COLOR]

label_het = "Heterosexual (Control)"
label_gay = "Homosexual (Treated)"
order_hetgay = [label_het,
                label_gay]

SINGLE_COLOR="#b3c6ff"
BINNED_COLOR="#b3c6ff"

filter_years = [2014,2015,2016,2017,2018,2019,2020,2021,2022]

## FUNCTIONS
def melt_groups(df, cols, id_vars=["name","code","year","treated"]):
    # Example: melt_groups(cases, ["hiv_m","hiv_f"])
    return (df.melt(id_vars=id_vars,
               value_vars=cols,
               var_name="group")
          .rename(columns={"variable":"code",
                           "value":"cases"}))

def binned_data(df, var, dict_hist):
    "returns the data in bins to be used in graphs"
    labels = []
    values = []
    for label,interval in dict_hist.items():
        start = interval[0]
        end = interval[1]
        query = f"{var} >= {start} & {var} <= {end}"
        labels.append(label)
        value = df[label] = df.query(query).shape[0]
        values.append(value)

    return pd.DataFrame({"labels":labels, "values":values})


## COMPUTE TREATMENT INTENSITY
def compute_treatment_intensity(df):
    df = df.set_index("year")
    baseline_het = df.loc[2018]["hiv_het"]
    baseline_h = df.loc[2018]["hiv_h"]
    baseline_f = df.loc[2018]["hiv_f"]
    baseline_hep_b = df.loc[2018]["hep_b_total"]
    baseline_hep_c = df.loc[2018]["hep_c_total"]

    df.loc[2019:2022,"diff_to_baseline_h"] = (df.loc[2019:2022]["hiv_h"] - 
                                                        baseline_h)
    df.loc[2019:2022,"pct_chg_to_baseline_h"]   = ((df.loc[2019:2022]["hiv_h"] - 
                                                     baseline_h)/baseline_h)

    df.loc[2019:2020,"diff_to_baseline_hep_b"] = (df.loc[2019:2020]["hep_b_total"] - 
                                                    baseline_hep_b)
    df.loc[2019:2020,"diff_to_baseline_hep_c"] = (df.loc[2019:2020]["hep_c_total"] - 
                                                    baseline_hep_c)

    df.loc[2019:2020,"pct_chg_to_baseline_hep_b"] = ((df.loc[2019:2020]["hep_b_total"] 
                                                     - baseline_hep_b)/baseline_hep_b)
    df.loc[2019:2020,"pct_chg_to_baseline_hep_c"] = ((df.loc[2019:2020]["hep_c_total"] 
                                                     - baseline_hep_c)/baseline_hep_c)
    df.loc[2019:2020,"diff_treat_control_hep_b"] = (df.loc[2019:2020]
                                                    ["pct_chg_to_baseline_h"] 
                                                      - df.loc[2019:2020]
                                                      ["pct_chg_to_baseline_hep_b"])
    df.loc[2019:2020,"diff_treat_control_hep_c"] = (df.loc[2019:2020]
                                                    ["pct_chg_to_baseline_h"] 
                                                  - df.loc[2019:2020]
                                                    ["pct_chg_to_baseline_hep_c"])


    df.loc[2019:2022,"diff_to_baseline_het"] = (df.loc[2019:2022]["hiv_het"] - 
                                                    baseline_het)
    df.loc[2019:2022,"pct_chg_to_baseline_het"] = ((df.loc[2019:2022]["hiv_het"] - 
                                                   baseline_het)/baseline_het)
    df.loc[2019:2022,"diff_to_baseline_f"] = df.loc[2019:2022]["hiv_f"] - baseline_f
    df.loc[2019:2022,"pct_chg_to_baseline_f"] = ((df.loc[2019:2022]["hiv_f"] - 
                                                 baseline_f)/baseline_f)

    df.loc[2019:2022,"diff_treat_control"] = (df.loc[2019:2022]
            ["pct_chg_to_baseline_h"] - df.loc[2019:2022]["pct_chg_to_baseline_het"])
    df.loc[2019:2022,"diff_treat_control_f"] = (df.loc[2019:2022]
                                        ["pct_chg_to_baseline_h"] - df.loc[2019:2022]
                                                ["pct_chg_to_baseline_f"])

    df["cummulative_prep"] = 0
    df.loc[2018,"cummulative_prep"] = df.loc[2018,"new_users"]/2

    for y in range(2019,2023):
        df.loc[y,"cummulative_prep"] = (df.loc[y-1,"cummulative_prep"] + 
                                          df.loc[y-1,"new_users"] + 
                                          df.loc[y,"new_users"]/2)

    df["treatment_intensity"] = df["cummulative_prep"]/baseline_h
    df = df.replace(np.inf, None)
    df = df.replace(-np.inf, None)
    return df

## SAVE TO STATA
cases = pd.read_csv("data/cases_cleaned.csv")
cases = cases.sort_values(['code','year'])
cases = cases.set_index(['code','year','name'])

cities = pd.read_csv("data/cities.csv")
cities = cities.sort_values(['code','year'])
cities = cities.set_index(['code'])#,'year','name'
cities["hiv_het"] = cities["hiv_m"] - cities["hiv_h"]
cities = cities.query("state_code != 42") # missing data for Santa Catarina

regions = pd.read_csv("data/regions.csv")
regions = regions.sort_values(['region_code','year'])
regions = regions.set_index(['region_code'])#,'year','name'
regions["hiv_het"] = regions["hiv_m"] - regions["hiv_h"]
regions = regions.query("state_code != 42") # missing data for Santa Catarina

index_city_18 = cities.query("year == 2018 & hiv_h > 5").index
cities_outcome = (cities.loc[index_city_18].groupby("code")
                    .apply(compute_treatment_intensity).reset_index())
cities_outcome = cities_outcome.set_index("code")

index_regions_18 = regions.query("year == 2018 & hiv_h > 5").index
regions_outcome = (regions.loc[index_regions_18].groupby("region_code")
                    .apply(compute_treatment_intensity).reset_index())
regions_outcome = regions_outcome.set_index("region_code")

## create thresholds
index_city_18_10 = cities_outcome.query("year == 2018 & hiv_h >= 10").index
index_city_18_15 = cities_outcome.query("year == 2018 & hiv_h >= 15").index
index_city_18_20 = cities_outcome.query("year == 2018 & hiv_h >= 20").index

cities_outcome["level_18_10"] = 0
cities_outcome["level_18_15"] = 0
cities_outcome["level_18_20"] = 0

cities_outcome.loc[index_city_18_10, "level_18_10"] = 1
cities_outcome.loc[index_city_18_15, "level_18_15"] = 1
cities_outcome.loc[index_city_18_20, "level_18_20"] = 1

index_regions_18_10 = regions.query("year == 2018 & hiv_h >= 10").index
index_regions_18_15 = regions.query("year == 2018 & hiv_h >= 15").index
index_regions_18_20 = regions.query("year == 2018 & hiv_h >= 20").index

regions_outcome["level_18_10"] = 0
regions_outcome["level_18_15"] = 0
regions_outcome["level_18_20"] = 0

regions_outcome.loc[index_regions_18_10, "level_18_10"] = 1
regions_outcome.loc[index_regions_18_15, "level_18_15"] = 1
regions_outcome.loc[index_regions_18_20, "level_18_20"] = 1

### DEFINE COHORTS
def query_treatment(df,year,thresh):
    return df.query(f"year == {year} & treatment_intensity >= {thresh}").index.unique()


def define_cohorts(df, thresh):
    treated_in_19 = query_treatment(df,2019,thresh)
    
    df_not_19 = cities_outcome.loc[~df.index.isin(treated_in_19)]
    treated_in_20 = query_treatment(df_not_19,2020,thresh)
    
    df_not_20 = df_not_19.loc[~df_not_19.index.isin(treated_in_20)]
    treated_in_21 = query_treatment(df_not_20,2021,thresh)
    
    df_not_21 = df_not_20.loc[~df_not_20.index.isin(treated_in_21)]
    treated_in_22 = query_treatment(df_not_21,2022,thresh)
    
    df[f"cohort_{thresh}"] = "Never"
    df.loc[treated_in_19, f"cohort_{thresh}"] = "2019"
    df.loc[treated_in_20, f"cohort_{thresh}"] = "2020"
    df.loc[treated_in_21, f"cohort_{thresh}"] = "2021"
    df.loc[treated_in_22, f"cohort_{thresh}"] = "2022"

thresh = 2
define_cohorts(cities_outcome, thresh)

thresh = 4
define_cohorts(cities_outcome, thresh)

define_cohorts(cities_outcome, 1)

(cities_outcome.reset_index()
    [["code","year","state_code","diff_treat_control","treatment_intensity",
      "hiv_het","hiv_h","level_18_10","level_18_15","level_18_20",
         "health_expend","gdpp","cash_transfer_expend","pop20_24",
         "covid_per_capita"]]
    .to_stata("data/cities_with_outcome.dta"))

(regions_outcome.reset_index()
    [["region_code","year","state_code","diff_treat_control","treatment_intensity",
      "hiv_het","hiv_h","level_18_10","level_18_15","level_18_20",
         "health_expend","gdpp","cash_transfer_expend","pop20_24",
         "covid_per_capita"]]
    .to_stata("data/regions_with_outcome.dta"))

## FIGURE 2.1
from numpy import count_nonzero
ax = sns.barplot(x="year", y="following",
                 data=cities.query("year in [2018,2019,2020,2021,2022]"),
                 estimator=np.sum, color=SINGLE_COLOR)
plt.title("Number of people taking the prophylaxis medication")
plt.xlabel("Year")
plt.ylabel("")

plt.savefig("graphs/chap_2_prep_brazil.png", 
               bbox_inches='tight', dpi=150)
plt.show()



## FIGURE 3.1
sns.displot(cities_outcome.reset_index().query("year >= 2019"),
             x="treatment_intensity",
             bins=np.linspace(0.01,30.1,25),
             col="year", col_wrap=2, height=1.5)
plt.ylabel("asdf")
#plt.title("Number of regions by treatment levels")
plt.savefig("graphs/chap_3_treat_by_year.png", 
               bbox_inches='tight', dpi=150)
plt.show()



## FIGURE 3.2
dict_hist = {
    "6-10": [6,10],
    "11-50":[11,50],
    "> 50": [50,100000]
}

het_count_hist = binned_data(cities.query("year == 2018"), "hiv_het", dict_hist)
gay_count_hist = binned_data(cities.query("year == 2018"), "hiv_h", dict_hist)

het_count_hist["group"] = label_het
gay_count_hist["group"] = label_gay


hist_count_cities = pd.concat([het_count_hist, gay_count_hist])

##### CITIES

fig, ax = plt.subplots(figsize=(8,3))
plt.subplot(1, 2, 1)
g = sns.barplot(hist_count_cities,
             x="labels", y="values", hue="group",
                palette=palette_hetgay,
                hue_order=order_hetgay)

plt.title("Cities: New HIV cases by group in 2018",fontsize=10)
sns.move_legend(g, "upper right", title="",fontsize=8)
plt.ylabel("Number of cities")
plt.xlabel("Number of cases by city")
plt.ylim([0,300])


##### REGION

plt.subplot(1, 2, 2)
het_count_hist = binned_data(regions.query("year == 2018"), "hiv_het", dict_hist)
gay_count_hist = binned_data(regions.query("year == 2018"), "hiv_h", dict_hist)

het_count_hist["group"] = label_het
gay_count_hist["group"] = label_gay

hist_count_region = pd.concat([het_count_hist, gay_count_hist])

#fig, ax = plt.subplots(figsize=(20,10))
g = sns.barplot(hist_count_region,
             x="labels", y="values", hue="group",
             palette=palette_hetgay,
             hue_order=order_hetgay)

plt.title("Regions: New HIV cases by group in 2018",fontsize=10)
sns.move_legend(g, "upper right", title="",fontsize=8)
plt.ylabel("Number of regions")
plt.xlabel(f"Number of cases by region")

plt.savefig("graphs/chap_3_new_cases_by_city.png", 
               bbox_inches='tight')
plt.ylim([0,300])
plt.show()

## FIGURE 3.3
#graph = cities.query(f"hiv_h > 1 & year in {filter_years}")
graph = cities_outcome.query("cohort_4.isin(['2019','2020']) ")

graph = melt_groups(graph.reset_index(),
                    ["hiv_het","hiv_h"], id_vars=["code","year","cohort_4"])

graph.loc[graph["group"] == "hiv_het","group"]  = label_het
graph.loc[graph["group"] == "hiv_h","group"]  = label_gay

g = sns.relplot(graph,
                x="year", y="cases", kind="line",
                 errorbar=None, hue="group", col="cohort_4",
                 height=3, aspect=1.5,
                 palette=palette_hetgay,
                 hue_order=order_hetgay)
plt.xticks(filter_years, fontsize=8)

titles = ["Cities with treatment intensity > 4 in 2021",
          "Cities with treatment intensity > 4 in 2022"]
for ax,title in zip(g.axes.flatten(),titles):
    ax.set_title(title )
    ax.axvline(x=2018, color='#990033', linestyle='-')

sns.move_legend(g, "lower center", title="",fontsize=8, ncol=2)
plt.savefig("graphs/chap_3_parallel_trends_treated_cities.png", 
               bbox_inches='tight')
plt.show()


## FIGURE 4.1
query = f"treatment_intensity.between(0.1,30) & level_18_15==1"
cases_scatter = cities_outcome.query(query)

plt.figure(figsize=(6, 4))
g = sns.scatterplot(cases_scatter, x="treatment_intensity", 
                y="diff_treat_control",
                size="hiv_h",sizes=(5, 300))

sns.move_legend(g, "lower right", title="New infections",fontsize=8, ncol=3)

plt.title("Treatment intensity x outcome variable")
plt.xlabel("Treatment intensity")
plt.ylabel("% of change in the outcome variable")

plt.savefig("graphs/chap4_cities_treatment_intensity.png", 
               bbox_inches='tight')
plt.show()