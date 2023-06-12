capture program drop prep_reg
program prep_reg
	syntax varlist(max=1) [if] [ , filename(string) controls(namelist)  absorb(namelist) cluster(namelist) ]
	preserve
	if `"`if'"' != "" {
		keep `if'
	}

	global outcome `1'
	global controls `controls'
	global fixed_effects `absorb'
	global clustervar `cluster'

	eststo clear
	estimates clear
	
	reghdfe $outcome treatment_intensity , a( $fixed_effects ) vce(cl `cluster' )
	estimates store model1

	reghdfe $outcome treatment_intensity $controls, a( $fixed_effects ) vce(cl `cluster' )
	estimates store model2

	reghdfe $outcome treatment_intensity covid_per_capita, a( $fixed_effects ) vce(cl `cluster' )
	estimates store model3
	
	reghdfe $outcome treatment_intensity  $controls covid_per_capita, a( $fixed_effects ) vce(cl `cluster' )
	estimates store model4
	

estfe . model*, labels()
	
esttab model*  using "`filename'", style(tex)  star(* 0.10 ** 0.05 *** 0.01) indicate("Controls=$controls" "Covid=covid_per_capita" ) drop(_cons 0.code 0.year) b(3) se(3) scalars(r2_a N)  label replace
estfe . model*, restore

	restore
end

cd /Users/rafaelfrade/arquivos/mestrado/tese_hiv/data/


use data/cities_with_outcome.dta, clear

eststo clear

// SUMMARY STATISTICS
label variable diff_treat_control "y: difference in new HIV cases"
label variable treatment_intensity "treatment intensity"
label variable health_expend "per capita expenditure in health"
label variable cash_transfer_expend "per capita expenditure in cash transfers"
label variable pop20_24 "population between 20 and 24 years"
label variable covid_per_capita "cases of covid per capita"


keep if diff_treat_control != .
estpost summarize diff_treat_control treatment_intensity  health_expend cash_transfer_expend pop20_24 covid_per_capita

esttab  using "tables/summary.tex", cells("n mean sd min max") label  b(%5.2f)


keep(diff_treat_control treatment_intensity hiv_het hiv_h health_expend cash_transfer_expend pop20_24 covid_per_capita)

global controls health_expend cash_transfer_expend pop20_24

gen y = diff_treat_control
prep_reg y if year >= 2019 , controls($controls) filename("tables/cities_5.tex") cluster(code) absorb(code year)

prep_reg y if year >= 2019 & level_18_15==1 , controls($controls) filename("tables/cities_15.tex") cluster(code) absorb(code year)




// REGIONS
use data/regions_with_outcome.dta, clear

gen code = region_code
gen y = diff_treat_control
prep_reg y if year >= 2019 , controls($controls) filename("tables/regions_5.tex") cluster(code) absorb(code year)

prep_reg y if year >= 2019 & level_18_15==1 , controls($controls) filename("tables/regions_15.tex") cluster(code) absorb(code year)



reg diff_treat_control treatment_intensity $controls if year == 2019 & level_18_10==1, cluster(code)
