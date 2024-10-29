# title-translations

title translations using AI

Run

```sparql
select (?e as ?book_id) (?wl as ?tibetan_title) {
  {
    ?e a :Work ;
       skos:prefLabel ?wl .
    FILTER(lang(?wl) = "bo-x-ewts")
  } union {
    ?e a :Instance .
    FILTER(not exists{
    	?e :instanceOf ?w .
        ?w skos:prefLabel ?wwl .
    })
    ?e skos:prefLabel ?wl .
    FILTER(lang(?wl) = "bo-x-ewts")
  }
} limit 10
```

and save in `tibetan_titles.csv`