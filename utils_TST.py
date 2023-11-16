from tqdm import tqdm
from transformers import (CONFIG_NAME, WEIGHTS_NAME)
import pandas as pd
import numpy as np
import random
import torch
import pandas
import json
import os
import time
import math
import copy
import nltk

# remove stop words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# TreebankWordDetokenizer: reverse word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('stopwords')
nltk.download('punkt')

relations = [
    'AtLocation', 'CapableOf', 'Causes', 'CausesDesire',
    'CreatedBy', 'DefinedAs', 'DesireOf', 'Desires', 'HasA',
    'HasFirstSubevent', 'HasLastSubevent', 'HasPainCharacter',
    'HasPainIntensity', 'HasPrerequisite', 'HasProperty',
    'HasSubevent', 'InheritsFrom', 'InstanceOf', 'IsA',
    'LocatedNear', 'LocationOfAction', 'MadeOf', 'MotivatedByGoal',
    'NotCapableOf', 'NotDesires', 'NotHasA', 'NotHasProperty',
    'NotIsA', 'NotMadeOf', 'PartOf', 'ReceivesAction', 'RelatedTo',
    'SymbolOf', 'UsedFor'
]

split_into_words = {
    'AtLocation': "at location",
    'CapableOf': "capable of",
    'Causes': "causes",
    'CausesDesire': "causes desire",
    'CreatedBy': "created by",
    'DefinedAs': "defined as",
    'DesireOf': "desire of",
    'Desires': "desires",
    'HasA': "has a",
    'HasFirstSubevent': "has first subevent",
    'HasLastSubevent': "has last subevent",
    'HasPainCharacter': "has pain character",
    'HasPainIntensity': "has pain intensity",
    'HasPrerequisite': "has prequisite",
    # actually it is "has prerequisite, but models were trained on it ..."
    'HasProperty': "has property",
    'HasSubevent': "has subevent",
    'InheritsFrom': "inherits from",
    'InstanceOf': 'instance of',
    'IsA': "is a",
    'LocatedNear': "located near",
    'LocationOfAction': "location of action",
    'MadeOf': "made of",
    'MotivatedByGoal': "motivated by goal",
    'NotCapableOf': "not capable of",
    'NotDesires': "not desires",
    'NotHasA': "not has a",
    'NotHasProperty': "not has property",
    'NotIsA': "not is a",
    'NotMadeOf': "not made of",
    'PartOf': "part of",
    'ReceivesAction': "receives action",
    'RelatedTo': "related to",
    'SymbolOf': "symbol of",
    'UsedFor': "used for",
    "/olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/olympics": "participate in olympics",
    "/film/film/music": "has music composed by",
    "/award/award_category/category_of": "is an award category of",
    "/music/performance_role/regular_performances./music/group_membership/role": None,
    "/location/country/capital": "capital",
    "/government/legislative_session/members./government/government_position_held/legislative_sessions": "members also attend",
    "/location/administrative_division/first_level_division_of": "is one of the first level administrtive division of",
    "/tv/tv_program/program_creator": "created by",
    "/base/americancomedy/celebrity_impressionist/celebrities_impersonated": "has impersonated",
    "/government/legislative_session/members./government/government_position_held/district_represented": "has members representing",
    "/education/educational_institution/students_graduates./education/education/student": "has graduated student",
    "/tv/tv_program/languages": "is spoken in",
    "/people/person/spouse_s./people/marriage/location_of_ceremony": "location of marriage ceremony",
    "/award/award_ceremony/awards_presented./award/award_honor/award_winner": "awarded to",
    "/government/politician/government_positions_held./government/government_position_held/basic_title": "has title in government",
    "/location/location/time_zones": "in time zone",
    "/film/person_or_entity_appearing_in_film/films./film/personal_film_appearance/type_of_appearance": "type of appearance in film",
    "/military/military_conflict/combatants./military/military_combatant_group/combatants": "has combatants",
    "/film/film_set_designer/film_sets_designed": "designed film set",
    "/medicine/symptom/symptom_of": "is a symptom of",
    "/language/human_language/countries_spoken_in": "is spoken in",
    "/base/popstra/celebrity/dated./base/popstra/dated/participant": "dated with",
    "/people/person/place_of_birth": "is born in",
    "/organization/organization/headquarters./location/mailing_address/country": "is located in",
    "/education/educational_institution/students_graduates./education/education/major_field_of_study": "major field of study",
    "/music/group_member/membership./music/group_membership/group": "is a member of",
    "/music/performance_role/regular_performances./music/group_membership/group": "played by",
    "/base/x2010fifaworldcupsouthafrica/world_cup_squad/current_world_cup_squad./base/x2010fifaworldcupsouthafrica/current_world_cup_squad/current_club": "from club",
    "/location/statistical_region/religions./location/religion_percentage/religion": "has main religion",
    "/education/educational_institution_campus/educational_institution": "educational institution",
    "/award/award_winning_work/awards_won./award/award_honor/honored_for": "won same awards with",
    "/film/film/produced_by": "is produced by",
    "/award/award_winning_work/awards_won./award/award_honor/award_winner": "earn awards to",
    "/people/profession/specialization_of": "is a specialization of",
    "/film/film/edited_by": "is edited by",
    "/film/film/release_date_s./film/film_regional_release_date/film_release_distribution_medium": "distributed by",
    "/location/statistical_region/gdp_real./measurement_unit/adjusted_money_value/adjustment_currency": "gdp measured by adjustment currency",
    "/sports/sports_position/players./american_football/football_historical_roster_position/position_s": "both football positions",
    "/location/location/partially_contains": "partially contains",
    "/film/film/story_by": "story by",
    "/base/popstra/location/vacationers./base/popstra/vacation_choice/vacationer": "has vacationer",
    "/base/localfood/seasonal_month/produce_available./base/localfood/produce_availability/seasonal_months": None,
    "/baseball/baseball_team/team_stats./baseball/baseball_team_stats/season": "participate in baseball season",
    "/music/instrument/instrumentalists": "played by",
    "/government/governmental_body/members./government/government_position_held/legislative_sessions": "take part in",
    "/organization/organization/headquarters./location/mailing_address/citytown": "headquarter located in",
    "/location/statistical_region/rent50_2./measurement_unit/dated_money_value/currency": None,
    "/sports/sports_position/players./sports/sports_team_roster/team": None,
    "/influence/influence_node/influenced_by": "is influenced by",
    "/food/food/nutrients./food/nutrition_fact/nutrient": "has nutrient",
    "/film/film/film_festivals": "is shown in festival",
    "/music/artist/track_contributions./music/track_contribution/role": "contribute to track as a role of",
    "/award/award_category/disciplines_or_subjects": "is an award of the discipline of",
    "/film/film_distributor/films_distributed./film/film_film_distributor_relationship/film": "distributed film",
    "/medicine/disease/notable_people_with_this_condition": "notable people with this condition",
    "/tv/tv_program/country_of_origin": "originated in",
    "/organization/organization/headquarters./location/mailing_address/state_province_region": "located in",
    "/sports/sports_position/players./sports/sports_team_roster/position": None,
    "/location/hud_foreclosure_area/estimated_number_of_mortgages./measurement_unit/dated_integer/source": "number of mortgages measured by",
    "/film/film/other_crew./film/film_crew_gig/film_crew_role": "take the role of",
    "/people/deceased_person/place_of_death": "place of death",
    "/base/marchmadness/ncaa_basketball_tournament/seeds./base/marchmadness/ncaa_tournament_seed/team": "has seed team",
    "/user/jg/default_domain/olympic_games/sports": "includes sports",
    "/location/hud_county_place/place": None,
    "/film/film/distributors./film/film_film_distributor_relationship/region": "distributed in region",
    "/music/genre/artists": "played by artists",
    "/film/film/costume_design_by": "costume designed by",
    "/people/ethnicity/geographic_distribution": "geographically distributed in",
    "/organization/endowed_organization/endowment./measurement_unit/dated_money_value/currency": "endowment measured by currency",
    "/people/person/gender": "of gender",
    "/base/eating/practicer_of_diet/diet": "practicer of diet",
    "/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_language": "serve with language",
    "/film/film/release_date_s./film/film_regional_release_date/film_regional_debut_venue": "premiere at",
    "/music/artist/origin": "comes from",
    "/influence/influence_node/peers./influence/peer_relationship/peers": "is in peer relationship with",
    "/sports/sports_league_draft/picks./sports/sports_league_draft_pick/school": "has attendee",
    "/award/award_category/winners./award/award_honor/award_winner": "awarded to",
    "/olympics/olympic_games/participating_countries": "participating countries",
    "/film/film/production_companies": "produced by companies",
    "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician": "has member",
    "/education/university/domestic_tuition./measurement_unit/dated_money_value/currency": "domestic tuition measured by currency",
    "/film/film/other_crew./film/film_crew_gig/crewmember": "has pther crew member",
    "/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/contact_category": "category of contact",
    "/award/award_nominee/award_nominations./award/award_nomination/award_nominee": "has same award nomination as ",
    "/film/film/dubbing_performances./film/dubbing_performance/actor": "dubbed by",
    "/location/country/form_of_government": "form of government",
    "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency": "net worth measured by currency",
    "/film/film/distributors./film/film_film_distributor_relationship/film_distribution_medium": "film distributed in the form of",
    "/base/biblioness/bibs_location/state": None,
    "/sports/sports_team/roster./american_football/football_historical_roster_position/position_s": "has position",
    "/location/capital_of_administrative_division/capital_of./location/administrative_division_capital_relationship/administrative_division": "capital of",
    "/music/record_label/artist": "owns artist",
    "/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics": "won olympic medals in",
    "/business/business_operation/assets./measurement_unit/dated_money_value/currency": "assets measured by currency",
    "/film/film/written_by": "written by",
    "/film/film/release_date_s./film/film_regional_release_date/film_release_region": "released in",
    "/tv/tv_writer/tv_programs./tv/tv_program_writer_relationship/tv_program": "writer tv program",
    "/film/film/film_production_design_by": "film production designed by",
    "/travel/travel_destination/how_to_get_here./travel/transportation/mode_of_transportation": "transportation to get there",
    "/sports/sports_team/roster./baseball/baseball_roster_position/position": "has position",
    "/award/award_nominee/award_nominations./award/award_nomination/award": "is nominated for",
    "/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor": "has an actor",
    "/base/popstra/celebrity/canoodled./base/popstra/canoodled/participant": "canoodled with",
    "/common/topic/webpage./common/webpage/category": "category of webpage",
    "/location/country/second_level_divisions": "is one of the second level administrtive division of",
    "/military/military_combatant/military_conflicts./military/military_combatant_group/combatants": "has military conflicts with",
    "/award/award_category/winners./award/award_honor/ceremony": "is awarded in ceremony",
    "/celebrities/celebrity/celebrity_friends./celebrities/friendship/friend": "is a friend of",
    "/tv/non_character_role/tv_regular_personal_appearances./tv/tv_regular_personal_appearance/person": "role taken by",
    "/sports/sports_league/teams./sports/sports_league_participation/team": "has participants",
    "/olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/olympics": "is a sport in",
    "/user/tsegaran/random/taxonomy_subject/entry./user/tsegaran/random/taxonomy_entry/taxonomy": None,
    "/base/petbreeds/city_with_dogs/top_breeds./base/petbreeds/dog_city_relationship/dog_breed": "has top dog breed",
    "/education/educational_degree/people_with_this_degree./education/education/student": "people with this degree",
    "/business/business_operation/industry": "main business",
    "/education/university/fraternities_and_sororities": "has fraternities and sororities",
    "/people/person/places_lived./people/place_lived/location": "lived in",
    "/dataworld/gardening_hint/split_to": None,
    "/organization/role/leaders./organization/leadership/organization": "is a leader position in",
    "/music/performance_role/guest_performances./music/recording_contribution/performance_role": "is a performance role in",
    "/base/aareas/schema/administrative_area/capital": "has capital",
    "/music/genre/parent_genre": "is genre of",
    "/people/person/spouse_s./people/marriage/type_of_union": "has type of union with spouse",
    "/soccer/football_player/current_team./sports/sports_team_roster/team": "is a soccer player in",
    "/sports/sports_team_location/teams": "has team",
    "/people/cause_of_death/people": "causes death to",
    "/location/location/adjoin_s./location/adjoining_relationship/adjoins": "adjoins",
    "/business/business_operation/operating_income./measurement_unit/dated_money_value/currency": "has income type of",
    "/government/government_office_category/officeholders./government/government_position_held/jurisdiction_of_office": "is government position in",
    "/film/film/featured_film_locations": "has filming locations in",
    "/olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country": "has atheletes from",
    "/base/aareas/schema/administrative_area/administrative_area_type": "has administrative type of",
    "/film/film/personal_appearances./film/personal_film_appearance/person": "has appearances of",
    "/sports/sports_team/colors": "has color in",
    "/base/biblioness/bibs_location/country": "locates in",
    "/film/film/estimated_budget./measurement_unit/dated_money_value/currency": "has value in the currency of",
    "/education/educational_institution/colors": "has color in",
    "/award/hall_of_fame/inductees./award/hall_of_fame_induction/inductee": "has inductee",
    "/government/politician/government_positions_held./government/government_position_held/legislative_sessions": "has politic position in",
    "/film/actor/film./film/performance/film": "acts in",
    "/award/award_winner/awards_won./award/award_honor/award_winner": "wins the same award with",
    "/education/university/local_tuition./measurement_unit/dated_money_value/currency": "has tuition measured by",
    "/base/popstra/celebrity/breakup./base/popstra/breakup/participant": "breaks up with",
    "/time/event/instance_of_recurring_event": "is an instance of",
    "/people/person/profession": "has profession in",
    "/education/field_of_study/students_majoring./education/education/student": "is a major of",
    "/user/ktrueman/default_domain/international_organization/member_states": "has member",
    "/music/instrument/family": "belongs to the musical instrument family of",
    "/ice_hockey/hockey_team/current_roster./sports/sports_team_roster/position": "has position type of",
    "/people/person/languages": "can speak",
    "/base/locations/continents/countries_within": "contains the country",
    "/music/artist/contribution./music/recording_contribution/performance_role": "contributes to the musical performance role of",
    "/government/politician/government_positions_held./government/government_position_held/jurisdiction_of_office": "has political power to govern",
    "/people/marriage_union_type/unions_of_this_type./people/marriage/location_of_ceremony": "is held in",
    "/education/educational_institution/campuses": "is an institution of",
    "/organization/organization/child./organization/organization_relationship/child": "has a children organization",
    "/sports/sports_team/sport": "is a team in",
    "/location/statistical_region/places_exported_to./location/imports_and_exports/exported_to": "exports to",
    "/award/award_nominated_work/award_nominations./award/award_nomination/nominated_for": "win awards at the same time with",
    "/soccer/football_team/current_roster./soccer/football_roster_position/position": "has position in",
    "/film/actor/film./film/performance/special_performance_type": "performs in the type of",
    "/award/award_ceremony/awards_presented./award/award_honor/honored_for": "gives award honor to",
    "/organization/organization_member/member_of./organization/organization_membership/organization": "is a member of",
    "/sports/sport/pro_athletes./sports/pro_sports_played/athlete": "has previous athlete",
    "/sports/pro_athlete/teams./sports/sports_team_roster/team": "previously played in",
    "/time/event/locations": "located in",
    "/tv/tv_personality/tv_regular_appearances./tv/tv_regular_personal_appearance/program": "appears in the program of",
    "/business/business_operation/revenue./measurement_unit/dated_money_value/currency": "operates business in the currency of",
    "/people/person/employment_history./business/employment_tenure/company": "worked in",
    "/location/statistical_region/gdp_nominal./measurement_unit/dated_money_value/currency": "measures gdp by the currency of",
    "/soccer/football_team/current_roster./sports/sports_team_roster/position": "plays in the position of",
    "/film/film/language": "has film language",
    "/location/statistical_region/gdp_nominal_per_capita./measurement_unit/dated_money_value/currency": "measures per capita gdp by the currency of",
    "/olympics/olympic_games/sports": "had sports",
    "/location/statistical_region/gni_per_capita_in_ppp_dollars./measurement_unit/dated_money_value/currency": "measures per capital ppp gdp by the currency of",
    "/education/field_of_study/students_majoring./education/education/major_field_of_study": "is a major as well as",
    "/medicine/disease/risk_factors": "has risk factor of",
    "/film/film/cinematography": "is cinematographed by",
    "/tv/tv_program/genre": "is genre of",
    "/film/film/genre": "is genre of",
    "/film/film/country": "is produced from",
    "/award/award_winning_work/awards_won./award/award_honor/award": "receives the award",
    "/film/film/executive_produced_by": "has executive producer",
    "/people/person/sibling_s./people/sibling_relationship/sibling": "is a sibling of",
    "/film/actor/dubbing_performances./film/dubbing_performance/language": "dubs by the language of",
    "/tv/tv_producer/programs_produced./tv/tv_producer_term/producer_type": "is a",
    "/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/medal": "won",
    "/sports/professional_sports_team/draft_picks./sports/sports_league_draft_pick/draft": "participated in",
    "/film/film/runtime./film/film_cut/film_release_region": "is released in",
    "/film/film/prequel": "has prequel",
    "/sports/sports_team/roster./american_football/football_roster_position/position": "has football position in",
    "/education/university/international_tuition./measurement_unit/dated_money_value/currency": "measures its tuition by the currency of",
    "/music/performance_role/track_performances./music/track_contribution/role": "plays the role of",
    "/location/country/official_language": "has official language",
    "/tv/tv_program/tv_producer./tv/tv_producer_term/producer_type": "has the producer type of",
    "/music/group_member/membership./music/group_membership/role": "plays the role in",
    "/tv/tv_network/programs./tv/tv_network_duration/program": "has tv program",
    "/people/deceased_person/place_of_burial": "is buried in",
    "/tv/tv_producer/programs_produced./tv/tv_producer_term/program": "produces",
    "/base/aareas/schema/administrative_area/administrative_parent": "is a part of",
    "/base/saturdaynightlive/snl_cast_member/seasons./base/saturdaynightlive/snl_season_tenure/cast_members": "is a cast member in the time with",
    "/broadcast/content/artist": "has the artist",
    "/award/award_category/nominees./award/award_nomination/nominated_for": "has nominees",
    "/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_location": "has service location in",
    "/film/film/film_format": "is a format of",
    "/education/educational_institution/school_type": "is a type of",
    "/education/educational_degree/people_with_this_degree./education/education/major_field_of_study": "has a field in",
    "/olympics/olympic_games/medals_awarded./olympics/olympic_medal_honor/medal": "has medal honor of",
    "/people/ethnicity/people": "is the ethnicity of",
    "/award/award_nominee/award_nominations./award/award_nomination/nominated_for": "win award by",
    "/location/location/contains": "contains",
    "/people/ethnicity/languages_spoken": "speaks",
    "/base/popstra/celebrity/friendship./base/popstra/friendship/participant": "is a friend of",
    "/location/hud_county_place/county": "is a county in",
    "/sports/sports_team/roster./basketball/basketball_roster_position/position": "has basketball position in",
    "/sports/professional_sports_team/draft_picks./sports/sports_league_draft_pick/school": "picks a player from",
    "/film/special_film_performance_type/film_performance_type./film/performance/film": "is the type of",
    "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity": "has a romantic relationship with",
    "/location/us_county/county_seat": "has a county seat of",
    "/organization/non_profit_organization/registered_with./organization/non_profit_registration/registering_agency": "is registered with",
    "/base/culturalevent/event/entity_involved": "involves",
    "/american_football/football_team/current_roster./sports/sports_team_roster/position": "has football position in",
    "/people/person/nationality": "is a person of",
    "/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month": "can be traveled in",
    "/film/film_subject/films": "is the subject of",
    "/award/ranked_item/appears_in_ranked_lists./award/ranking/list": "gets rank in the list of",
    "/people/person/spouse_s./people/marriage/spouse": "has a marriage with",
    "/user/alexander/philosophy/philosopher/interests": "has interests in",
    "/location/administrative_division/country": "is a part of",
    "/business/job_title/people_with_this_title./business/employment_tenure/company": "is a position in",
    "/film/film/film_art_direction_by": "has art director",
    "/media_common/netflix_genre/titles": "is the genre of",
    "/organization/organization_founder/organizations_founded": "founded",
    "/organization/organization/place_founded": "is founded in",
    "/people/person/religion": "has the belief of",
    "/education/educational_degree/people_with_this_degree./education/education/institution": "is a degree in",
    "/film/director/film": "directs"
}

relations_atomic = ["oReact", "oEffect", "oWant", "xAttr", "xEffect", "xIntent", \
"xNeed", "xReact", "xWant"]
split_into_words_atomic = {
    '<oReact>': 'As a result, others feel',
    '<oEffect>': 'Others then',
    # TODO: some value here might not contain 'to' but need 'to'
    '<oWant>': 'As a result, others want',
    '<xAttr>': 'This person is seen as',
    '<xEffect>': 'This person then',
    # TODO: some value here might not contain 'to' but need 'to'
    '<xIntent>': 'Because this person wanted',
    '<xNeed>': 'Before, this person needed',
    '<xReact>': 'As a result, this person feels',
    '<xWant>': 'As a result, this person wants'
}


def load_conceptnet_noCase(dataset_path=None,
                        cls_token=None,
                        eos_token=None,
                        sep_token=None,
                        rel_lang=True,
                        toy=False,
                        discard_negative=True,
                        sep=False,
                        add_sep=False,
                        prefix=None,
                        model_type=None):
    if not eos_token:
        end_token = ""
    with open(dataset_path, encoding='utf_8') as f:
        f = f.read().splitlines()
        if toy:
            # no shuffle
            # raise Exception
            random.shuffle(f)
            f = f[:1000]
            print("Warning: toy experiment")
        output = []
        # for line in tqdm(f):
        for id_line, line in enumerate(f):
            try:
                rel, e1, e2, label = line.split("\t")
            except:
                print(line.split("\t"))
                raise ValueError
            if discard_negative and label == "0":
                continue
            if not discard_negative:
                # convert to int, to avoid being encoded
                try:
                    label = int(label)
                except:
                    # in ConceptNet training data the label is float
                    label = -1
            # e1
            if 'bert' in model_type:
                e1 += (" " + sep_token)
                e1 = cls_token + " " + e1
            else:
                if add_sep:
                    e1 += (" " + sep_token)
                if prefix:
                    e1 = prefix + " " + e1
            # e2
            e2 += (" " + eos_token)
            # rel
            if rel_lang:
                rel = split_into_words[rel]
                if not rel:
                    continue
            else:
                rel = rel.lower()
            if 'bert' in model_type:
                rel += (" " + sep_token)
            elif add_sep:
                rel += (" " + sep_token)
            # [CLS] e1 [SEP] rel [SEP] e2 [EOS]
            output.append((e1, rel, e2, label, id_line))
        # print some samples for sure
        print(output[-3:])
    return output


## index builder functions
def load_conceptnet_withCase_withMidPrompt(args,
                        dataset_path=None,
                        cases_path=None,
                        cls_token=None,
                        eos_token=None,
                        sep_token=None,
                        rel_lang=True,
                        toy=False,
                        discard_negative=True,
                        sep=False,
                        add_sep=False,
                        prefix=None,
                        model_type=None,
                        if_without_case=False,
                        num_cases=3):
    if not eos_token:
        end_token = ""
    if not args.if_without_case:
        with open(cases_path, encoding='utf_8') as case_f:
            case_lines = case_f.readlines()
    with open(dataset_path, encoding='utf_8') as f:
        f = f.read().splitlines()
        # # 2400
        # print("len(f): ", len(f))
        # # 1200
        # print("len(case_lines): ", len(case_lines))
        # this should hold in most cases (except for using dev2)
        if not args.if_without_case:
            # case_lines_cnter
            if 'dev1' in dataset_path.split('/')[-1]:
                case_lines_cnter = 0
            elif  'dev2' in dataset_path.split('/')[-1]:
                case_lines_cnter = 600
            else:
                case_lines_cnter = 0

        output = []
        # for line in tqdm(f):
        for id_line, line in enumerate(f):
            rel, e1, e2, label = line.split("\t")
            if discard_negative and label == "0":
                continue
            if not discard_negative:
                # convert to int, to avoid being encoded
                try:
                    label = int(label)
                except:
                    # in ConceptNet training data the label is float
                    label = -1

            if not args.if_without_case:
                # label's positive, begin collect cases
                # cases: rel\te1\te2\t\t..
                cases = case_lines[case_lines_cnter].strip('\n')
                cases = cases.split('\t\t')
                # Q: restrict number of cases to use
                assert len(cases) == num_cases
                if len(cases) > num_cases:
                    cases = cases[0:num_cases]
                case_sents = []
                for id_case, case in enumerate(cases):
                    case = case.split('\t')
                    # print('case')
                    # print(case)
                    case_rel = case[0]
                    # if no cases for current data
                    if case_rel == '':
                        break
                    if rel_lang:
                        case_rel = split_into_words[case_rel]
                        if not case_rel:
                            raise Exception
                    else:
                        case_rel = case_rel.lower()
                        # can't be 0 or 1
                        assert args.dataset_selection > 1
                    if args.use_special_tokens_to_split_retrieved_cases:
                        case[1] += "<split_source/target>"
                        case[2] += "<split_cases>"
                    if args.if_only_use_relation_and_retrieved_target:
                        case = ' '.join([case_rel, case[2]])
                    elif args.if_only_use_retrieved_target:
                        case = case[2]
                    else:
                        case = ' '.join([case[1], case_rel, case[2]])
                    case_sents.append(case)
                if args.if_with_strt_mid_promp and not args.if_without_case:
                    strt_prompt = 'Here are some similar cases to infer from: '
                    mid_prompt ='With the similar cases we can infer that: '
                    case_sents = strt_prompt + ' '.join(case_sents) + mid_prompt
                else:
                    case_sents = ' '.join(case_sents)
                # case_sents = ' '.join(case_sents) + " " + sep_token

            # e2
            # e2 <eos>
            e2 += (" " + eos_token)
            # e1
            if args.use_special_tokens_to_split_retrieved_cases:
                e1 += (" " + "<split_source/target>")
            # rel
            if rel_lang:
                if rel in split_into_words:
                    rel = split_into_words[rel]
                else:
                    print('rel:', rel)
                    rel = rel
                if not rel:
                    print(id_line, rel, e1, e2)
                    raise Exception
            else:
                rel = rel.lower()
                raise Exception
            # COMET baseline
            if if_without_case:
                # case_sents = '[SEP]'
                case_sents = " "
                # only print once
                if id_line == 0:
                    print('INFO: No cases are allowed to be used')
            output.append((case_sents, e1, rel, e2, label, id_line))
            if not args.if_without_case:
                case_lines_cnter += 1
        # print some samples for sure
        print(output[-3:])
    return output


def zipped_flatten(outer):
    return [(key, fill, el) for key, fill, inner in outer for el in inner]


# add if_without_none
def load_data_atomic(dataset_path, if_without_none=False):
    data = []
    print("Loading data from ", dataset_path)
    df = pandas.read_csv(dataset_path, index_col=0)
    df.iloc[:, :9] = df.iloc[:, :9].apply(
        lambda col: col.apply(json.loads))
    cat_len_noter = []
    for cat in relations_atomic:
        attr = df[cat]
        # print(len(attr))
        tmp_rel_data = zipped_flatten(zip(
            attr.index, ["<{}>".format(cat)] * len(attr), attr.values))
        if if_without_none:
            tmp_rel_data_without_none = []
            for id_line, line in enumerate(tmp_rel_data):
                e1, rel, e2 = line
                if "none" in e2.lower():
                    continue
                tmp_rel_data_without_none.append(line)
            tmp_rel_data = tmp_rel_data_without_none
        data += tmp_rel_data
        cat_len_noter.append(len(data))
    # TOCHECK: not same number of data?
    print('info', len(df), len(data))
    print('atomic_data: ', data[0])
    return data, cat_len_noter


def load_atomic_noCase(dataset_path=None,
                        cls_token=None,
                        eos_token=None,
                        sep_token=None,
                        rel_lang=True,
                        add_sep=False,
                        prefix=None,
                        model_type=None):
    print(dataset_path)
    loaded_data, _ = load_data_atomic(dataset_path)
    if not eos_token:
        end_token = ""
    # with open(dataset_path, encoding='utf_8') as f:
        # f = f.read().splitlines()
    output = []
    # for line in tqdm(f):
    for id_line, line in enumerate(loaded_data):
        e1, rel, e2 = line
        # all label for atomic is 1, to be in the same form with conceptnet
        label = 1
        # e1
        if 'bert' in model_type:
            e1 += (" " + sep_token)
            e1 = cls_token + " " + e1
        else:
            if add_sep:
                e1 += (" " + sep_token)
            if prefix:
                e1 = prefix + " " + e1
        # e2
        e2 += (" " + eos_token)
        # rel
        if rel_lang:
            rel = split_into_words_atomic[rel]
            if not rel:
                raise Exception
                continue
        else:
            rel = rel.lower()
        if 'bert' in model_type:
            rel += (" " + sep_token)
        elif add_sep:
            rel += (" " + sep_token)
        # [CLS] e1 [SEP] rel [SEP] e2 [EOS]
        output.append((e1, rel, e2, label, id_line))
    # print some samples for sure
    print(output[-3:])
    return output


def load_atomic_withCase(dataset_path=None,
                        cases_path=None,
                        cls_token=None,
                        eos_token=None,
                        sep_token=None,
                        rel_lang=True,
                        add_sep=False,
                        prefix=None,
                        model_type=None,
                        if_without_case=False,
                        if_allow_same_sub=False):

    loaded_data, _ = load_data_atomic(dataset_path)
    if not eos_token:
        end_token = ""

    with open(cases_path, encoding='utf_8') as case_f:
        case_lines = case_f.readlines()
        output = []
        for id_line, line in enumerate(loaded_data):
            # label: to keep in same format with conceptnet
            label = 1
            e1, rel, e2 = line
            # label's positive, begin collect cases
            # cases: rel\te1\te2\t\t..
            cases = case_lines[id_line].strip('\n')
            cases = cases.split('\t\t')
            assert len(cases) == 5
            # Q:
            # cases = select_cases_from_given_cases_v2(cases, e1, rel, e2, if_allow_same_sub)
            # cases = cases[0:5]

            case_sents = []
            for id_case, case in enumerate(cases):
                case = case.split('\t')
                # print('case')
                # print(case)
                case_rel = case[0]
                # if no cases for current data
                if case_rel == '' or case_rel == '\n':
                    break
                if rel_lang:
                    if case_rel in split_into_words_atomic:
                        case_rel = split_into_words_atomic[case_rel]
                    else:
                        print('case_rel, case')
                        print(case_rel, case)
                        raise Exception
                    if not case_rel:
                        raise Exception
                else:
                    case_rel = case_rel.lower()
                    # raise Exception
                # add '.' to obj of an case that not ends with '.'
                if not case[2].strip().endswith('.'):
                    case[2] += '.'
                case = ' '.join([case[1] + '.', case_rel, case[2]])
                case_sents.append(case)
            case_sents = ' '.join(case_sents)
            # case_sents
            # e1 rel e2. e1 rel e2. ...[SEP]
            if 'bert' in model_type:
                # e1 += (" " + sep_token)
                # e1 = cls_token + " " + e1
                case_sents += (" " + sep_token)
                case_sents = cls_token + " " + case_sents
            else:
                if add_sep:
                    # e1 += (" " + sep_token)
                    case_sents += (" " + sep_token)
                else:
                    raise Exception
                if prefix:
                    # e1 = prefix + " " + e1
                    case_sents = prefix + " " + case_sents
                    raise Exception
            # e2
            # e2 <eos>
            e2 += (" " + eos_token)
            # e1
            e1 += '.'
            # rel
            if rel_lang:
                if rel in split_into_words_atomic:
                    rel = split_into_words_atomic[rel]
                else:
                    print('rel:', rel)
                    rel = rel
                    raise Exception
                if not rel:
                    print(id_line, rel, e1, e2)
                    raise Exception
            else:
                rel = rel.lower()
            # JUSTTRY:
            if if_without_case:
                case_sents = ''
                # only print once
                if id_line == 0:
                    print('INFO: No cases are allowed to be used')
            output.append((case_sents, e1, rel, e2, label, id_line))
        # print some samples for sure
        print(output[-3:])
    return output


def load_conceptnet_pure(dataset_path=None, rel_lang=True, discard_negative=True):
    with open(dataset_path, encoding='utf_8') as f:
        f = f.read().splitlines()
        output = []
        # for line in tqdm(f):
        for id_line, line in enumerate(f):
            rel, e1, e2, label = line.split("\t")
            # filter data instances whose label is not positive
            if discard_negative and label == "0":
                continue
            e1 = e1.strip()
            e2 = e2.strip()
            if e1.endswith('.'):
                e1 = e1[:-1]
            if e2.endswith('.'):
                e2 = e2[:-1]
                # print("e2 ends with '.'")
            if not discard_negative:
                # convert to int, to avoid being encoded
                try:
                    label = int(label)
                except:
                    # in ConceptNet training data the label is float
                    label = -1
            # rel
            if rel_lang:
                rel = split_into_words[rel]
                if not rel:
                    print("Warning: ", rel, " not found in dict split_into_words")
                    continue
            else:
                rel = rel.lower()
            # no special tokens added
            output.append((e1, rel, e2, label, id_line))
        # print some samples
        print("load_conceptnet_pure, output[-3:]: ", output[-3:])
    return output

# load_atomic_pure: not adding any special tokens
def load_atomic_pure(dataset_path=None, rel_lang=True, toy=False, if_without_none=False):
    if not toy:
        loaded_data, _ = load_data_atomic(dataset_path, if_without_none=if_without_none)
    else:
        # this loaded_data is sampled
        # with open('~/Data/try/sampled_atomic_loaded_data.json', 'r') as f:
        #     loaded_data = json.load(f)
        raise Exception("Toy data is used")

    output = []
    for id_line, line in enumerate(loaded_data):
        e1, rel, e2 = line
        # Newly added in 2/19/2021; do not need to predict '.', we added '.' for each additional case
        e1 = e1.strip()
        rel = rel.strip()
        e2 = e2.strip()
        if e1.endswith('.'):
            e1 = e1[:-1]
        if e2.endswith('.'):
            e2 = e2[:-1]
        # rel
        if rel_lang:
            rel = split_into_words_atomic[rel]
            if not rel:
                raise Exception
        else:
            rel = rel.lower()
        # all label for atomic is 1, to be in the same form with conceptnet
        label = 1
        # no special tokens
        output.append((e1, rel, e2, label, id_line))
    # print some samples for sure
    print("load_atomic_pure, output[-3:]:", output[-3:])
    return output

# output: train_datasets, eval_datasets, test_datasets
# train_datasets: [(e1, rel, e2, label, id_line), (), ...]
def load_shakespear(args, dataset_path):
    ori_dataset_path = os.path.join(dataset_path, 'Shakes_data')
    processed_dataset_path = os.path.join(dataset_path, 'Shakes_processed_lines')
    all_file_names = os.listdir(ori_dataset_path)
    # newly added
    all_file_names = sorted(all_file_names)
    train_datasets, eval_datasets, test_datasets = [], [], []
    # for retriever's usage
    train_lines, eval_lines, test_lines = [], [], []
    assert len(all_file_names) % 2 == 0 and len(all_file_names) > 0

    ## get lines and datasets
    for ith_poem in range(int(len(all_file_names)/2)):
        tmp_modern_file = all_file_names[2*ith_poem]
        tmp_origin_file = all_file_names[2*ith_poem+1]
        # same poem
        assert tmp_modern_file.split('_')[0] == tmp_origin_file.split('_')[0]
        # modern and original
        assert 'modern' in tmp_modern_file and 'original' in tmp_origin_file
        cur_lines = train_lines
        cur_datasets = train_datasets
        if 'twelfthnight' in tmp_modern_file:
            cur_lines = eval_lines
            cur_datasets = eval_datasets
        elif 'romeojuliet' in tmp_modern_file:
            cur_lines = test_lines
            cur_datasets = test_datasets
        len_cur_datasets = len(cur_datasets)
        # open modern file
        cnt_included_data_cur_file = 0
        with open(os.path.join(ori_dataset_path, tmp_modern_file), 'r') as fm, \
            open(os.path.join(ori_dataset_path, tmp_origin_file), 'r') as fo:
            cur_modern_lines = fm.readlines()
            cur_origin_lines = fo.readlines()
            assert len(cur_modern_lines) == len(cur_origin_lines) and len(cur_origin_lines) > 0
            for id_line in range(len(cur_modern_lines)):
                tmp_modern_line = cur_modern_lines[id_line].strip()
                tmp_origin_line = cur_origin_lines[id_line].strip()
                if tmp_modern_line == '':
                    assert tmp_origin_line == ''
                    print("Warning: tmp_modern_line is '', which is empty")
                    continue
                e1 = tmp_modern_line
                if args.if_use_relation_for_shakes:
                    rel = "Shakespeare's style is "
                else:
                    rel = ' '
                e2 = tmp_origin_line
                label = 1
                if len(e1) >= args.max_e1:
                    print("e1: {}, max_e1: {}".format(e1, args.max_e1))
                    continue
                if len(e2) >= args.max_e2:
                    print("e2: {}, max_e2: {}".format(e2, args.max_e2))
                    continue
                cnt_included_data_cur_file += 1
                cur_lines.append(rel + '\t' + e1 + '\t' + e2 + '\n')
                # cur_datasets.append((e1, rel, e2, label, len_cur_datasets + id_line))
                cur_datasets.append((e1, rel, e2, label, len_cur_datasets + cnt_included_data_cur_file))

    print("len(train_lines): {}, len(eval_lines): {}, len(test_lines): {}".format(len(train_lines), len(eval_lines), len(test_lines)))
    print("len(train_datasets): {}, len(eval_datasets): {}, len(test_datasets): {}".format(len(train_datasets), len(eval_datasets), len(test_datasets)))
    if 'train_lines.txt' not in os.listdir(processed_dataset_path):
        print('Creating train lines, eval lines and test lines')
        with open(os.path.join(processed_dataset_path, 'train_lines.txt'), 'w') as f:
            f.writelines(train_lines)
        with open(os.path.join(processed_dataset_path, 'eval_lines.txt'), 'w') as f:
            f.writelines(eval_lines)
        with open(os.path.join(processed_dataset_path, 'test_lines.txt'), 'w') as f:
            f.writelines(test_lines)
    return [train_datasets], [eval_datasets], [test_datasets]


# load and preprocess shakespeare data during generation
def load_shakes_withCase(args,
                        dataset_path=None,
                        cases_path=None,
                        cls_token=None,
                        eos_token=None,
                        sep_token=None,
                        rel_lang=True,
                        add_sep=False,
                        prefix=None,
                        model_type=None,
                        if_without_case=False,
                        if_allow_same_sub=False):

    _, _, loaded_data = load_shakespear(args, dataset_path=dataset_path)
    assert len(loaded_data) == 1
    loaded_data = loaded_data[0]
    # loaded_data, _ = load_data_atomic(dataset_path)
    if not eos_token:
        end_token = ""
    if not args.if_without_case:
        with open(cases_path, encoding='utf_8') as case_f:
            case_lines = case_f.readlines()
        if not len(case_lines) == len(loaded_data):
            print('len(case_lines): ', len(case_lines), 'len(loaded_data): ', len(loaded_data))
            raise Exception("len(case_lines) != len(loaded_data)")
        print('len(case_lines): ', len(case_lines), 'len(loaded_data): ', len(loaded_data))

    output = []
    for id_line, line in enumerate(loaded_data):
        # label: to keep in same format with conceptnet
        label = 1
        try:
            e1, rel, e2, label, data_id = line
        except:
            print("line: ", line)
            raise Exception
        # e2 <eos>
        e2 += (" " + eos_token)
        if args.use_special_tokens_to_split_retrieved_cases:
            e1 += (" " + "<split_source/target>")
        if not args.if_without_case:
            # label's positive, begin collect cases
            # cases: rel\te1\te2\t\t..
            cases = case_lines[id_line].strip('\n')
            cases = cases.split('\t\t')
            # len(cases) should equals num_cases
            assert len(cases) >= 1

            case_sents = []
            for id_case, case in enumerate(cases):
                case = case.split('\t')
                case_rel = case[0]
                # case[2] += ';'
                if args.use_special_tokens_to_split_retrieved_cases:
                    case[1] += "<split_source/target>"
                    case[2] += "<split_cases>"
                if args.if_only_use_relation_and_retrieved_target:
                    case = ' '.join([case_rel, case[2]])
                elif args.if_only_use_retrieved_target:
                    case = case[2]
                else:
                    case = ' '.join([case[1], case_rel, case[2]])
                case_sents.append(case)
            if args.if_with_strt_mid_promp and not args.if_without_case:
                strt_prompt = 'Here are some similar cases to infer from: '
                mid_prompt ='With the similar cases we can infer that: '
                case_sents = strt_prompt + ' '.join(case_sents) + mid_prompt
            else:
                case_sents = ' '.join(case_sents)
            case_sents += " " + sep_token
        else:
            # case_sents = sep_token
            case_sents = " "
            # only print once
            if id_line == 0:
                print('INFO: No cases are allowed to be used')
        output.append((case_sents, e1, rel, e2, label, id_line))
    # print some samples for sure
    print(output[-3:])
    return output


def load_e2e(args, dataset_path, data_type):
    if data_type == 'train':
        data = pd.read_csv(os.path.join(dataset_path, 'trainset.csv'))
        path_lines = os.path.join(dataset_path, 'train_lines.txt')
    elif data_type == 'eval':
        data = pd.read_csv(os.path.join(dataset_path, 'devset.csv'))
        path_lines = os.path.join(dataset_path, 'eval_lines.txt')
    elif data_type == 'test':
        data = pd.read_csv(os.path.join(dataset_path, 'testset_w_refs.csv'))
        path_lines = os.path.join(dataset_path, 'test_lines.txt')
    datasets, lines = [], []
    ref = data['ref']
    mr = data['mr']
    assert len(ref) == len(mr)
    print('len(lines): ', len(lines))
    print('len(mr): ', len(mr))
    for ith_table in range(len(mr)):
        if not ('\r\n' in mr[ith_table] or '\n' in mr[ith_table]):
            e1 = mr[ith_table]
        else:
            print("Warning: found '\\r\\n': mr[ith_table]: {}".format(mr[ith_table]))
            e1 = mr[ith_table].replace('\r\n', ' ')
            e1 = e1.replace('\n', ' ')
        rel = ' '
        if not ('\r\n' in ref[ith_table] or '\n' in ref[ith_table]):
            e2 = ref[ith_table]
        else:
            print("Warning: found '\\r\\n': ref[ith_table]: {}".format(ref[ith_table]))
            e2 = ref[ith_table].replace('\r\n', ' ')
            e2 = e2.replace('\n', ' ')
            print("e2: ", e2)
        label = 1
        id_line = ith_table
        lines.append(rel + '\t' + e1 + '\t' + e2 + '\n')
        datasets.append((e1, rel, e2, label, id_line))
    assert len(lines) == len(mr)

    # write lines
    with open(path_lines, 'w') as f:
        f.writelines(lines)
    # checking whether more lines are created
    with open(path_lines, 'r') as f:
        tmp_lines = f.readlines()
    try:
        assert len(tmp_lines) == len(lines)
    except:
        print('len(tmp_lines): ', len(tmp_lines))
        print('len(lines): ', len(lines))
        raise Exception
    return [datasets]



# load and preprocess e2e data during generation
def load_e2e_withCase(args,
                        dataset_path=None,
                        cases_path=None,
                        cls_token=None,
                        eos_token=None,
                        sep_token=None,
                        rel_lang=True,
                        add_sep=False,
                        prefix=None,
                        model_type=None,
                        if_without_case=False,
                        if_allow_same_sub=False):

    # _, _, loaded_data = load_shakespear(args, dataset_path=dataset_path)
    loaded_data = load_e2e(args, dataset_path, data_type='test')
    assert len(loaded_data) == 1
    loaded_data = loaded_data[0]
    # loaded_data, _ = load_data_atomic(dataset_path)
    if not eos_token:
        end_token = ""
    if not args.if_without_case:
        with open(cases_path, encoding='utf_8') as case_f:
            case_lines = case_f.readlines()
        if not len(case_lines) == len(loaded_data):
            print('len(case_lines): ', len(case_lines), 'len(loaded_data): ', len(loaded_data))
            raise Exception("len(case_lines) != len(loaded_data)")
        print('len(case_lines): ', len(case_lines), 'len(loaded_data): ', len(loaded_data))

    output = []
    for id_line, line in enumerate(loaded_data):
        # label: to keep in same format with conceptnet
        label = 1
        try:
            e1, rel, e2, label, data_id = line
        except:
            print("line: ", line)
            raise Exception
        # e2 <eos>
        e2 += (" " + eos_token)
        if args.use_special_tokens_to_split_retrieved_cases:
            e1 += (" " + "<split_source/target>")
        if not args.if_without_case:
            # label's positive, begin collect cases
            # cases: rel\te1\te2\t\t..
            cases = case_lines[id_line].strip('\n')
            cases = cases.split('\t\t')
            # len(cases) should equals num_cases
            assert len(cases) >= 1

            case_sents = []
            for id_case, case in enumerate(cases):
                case = case.split('\t')
                case_rel = case[0]
                # case[2] += ';'
                if args.use_special_tokens_to_split_retrieved_cases:
                    case[1] += "<split_source/target>"
                    case[2] += "<split_cases>"
                if args.if_only_use_relation_and_retrieved_target:
                    case = ' '.join([case_rel, case[2]])
                elif args.if_only_use_retrieved_target:
                    case = case[2]
                else:
                    case = ' '.join([case[1], case_rel, case[2]])
                case_sents.append(case)
            if args.if_with_strt_mid_promp and not args.if_without_case:
                strt_prompt = 'Here are some similar cases to infer from: '
                mid_prompt ='With the similar cases we can infer that: '
                case_sents = strt_prompt + ' '.join(case_sents) + mid_prompt
            else:
                case_sents = ' '.join(case_sents)
            case_sents += " " + sep_token
        else:
            # case_sents = sep_token
            case_sents = " "
            # only print once
            if id_line == 0:
                print('INFO: No cases are allowed to be used')
        output.append((case_sents, e1, rel, e2, label, id_line))
    # print some samples for sure
    print(output[-3:])
    return output


# For generation_Bart.py
def load_atomic_withCase_withMidPrompt(args,
                        dataset_path=None,
                        cases_path=None,
                        cls_token=None,
                        eos_token=None,
                        sep_token=None,
                        rel_lang=True,
                        add_sep=False,
                        prefix=None,
                        model_type=None,
                        if_allow_same_sub=False):

    loaded_data, _ = load_data_atomic(dataset_path)
    # if_without_none: get rid of knowledge tuple with "None" obj
    if args.if_without_none:
        accepted_loaded_data = []
        for id_line, line in enumerate(loaded_data):
            e1, rel, e2 = line
            if "none" in e2.lower():
                continue
            accepted_loaded_data.append(loaded_data[id_line])
    else:
        accepted_loaded_data = loaded_data

    if not eos_token:
        end_token = ""
    if not args.if_without_case:
        with open(cases_path, encoding='utf_8') as case_f:
            case_lines = case_f.readlines()
        if not len(case_lines) == len(accepted_loaded_data):
            print('len(case_lines): ', len(case_lines), 'len(accepted_loaded_data): ', len(accepted_loaded_data))
            raise Exception("len(case_lines) != len(accepted_loaded_data)")
        print('len(case_lines): ', len(case_lines), 'len(accepted_loaded_data): ', len(accepted_loaded_data))

    output = []
    for id_line, line in enumerate(accepted_loaded_data):
        # label: to keep in same format with conceptnet
        label = 1
        e1, rel, e2 = line
        e1 = e1.strip()
        rel = rel.strip()
        e2 = e2.strip()
        if e1.endswith('.'):
            e1 = e1[:-1]
        if e2.endswith('.'):
            e2 = e2[:-1]
        # e2 <eos>
        e2 += (" " + eos_token)
        if args.use_special_tokens_to_split_retrieved_cases:
            e1 += (" " + "<split_source/target>")
        # label's positive, begin collect cases
        # cases: rel\te1\te2\t\t..
        if not args.if_without_case:
            cases = case_lines[id_line].strip('\n')
            cases = cases.split('\t\t')
            # len(cases) should equals num_cases
            assert len(cases) >= 1

            case_sents = []
            for id_case, case in enumerate(cases):
                case = case.split('\t')
                case_rel = case[0]
                # if no cases for current data
                if case_rel == '' or case_rel == '\n':
                    raise Exception
                    # break
                if rel_lang:
                    if case_rel in split_into_words_atomic:
                        case_rel = split_into_words_atomic[case_rel]
                    else:
                        print('case_rel, case')
                        print(case_rel, case)
                        raise Exception
                    if not case_rel:
                        raise Exception
                else:
                    case_rel = case_rel.lower()
                    # raise Exception
                # add '.' to obj of an case that not ends with '.'
                if not case[1].strip().endswith('.'):
                    case[1] += '.'
                if not case[2].strip().endswith('.'):
                    case[2] += '.'
                # case[2] += ';'
                if args.use_special_tokens_to_split_retrieved_cases:
                    case[1] += "<split_source/target>"
                    case[2] += "<split_cases>"
                if args.if_only_use_relation_and_retrieved_target:
                    case = ' '.join([case_rel, case[2]])
                elif args.if_only_use_retrieved_target:
                    case = case[2]
                else:
                    case = ' '.join([case[1], case_rel, case[2]])
                case_sents.append(case)
            if args.if_with_strt_mid_promp and not args.if_without_case:
                strt_prompt = 'Here are some similar cases to infer from: '
                mid_prompt ='With the similar cases we can infer that: '
                case_sents = strt_prompt + ' '.join(case_sents) + mid_prompt
            else:
                case_sents = ' '.join(case_sents)
            # case_sents
            # e1 rel e2. e1 rel e2. ...[SEP]
            if 'bert' in model_type or 'dpr' in model_type:
                # e1 += (" " + sep_token)
                # e1 = cls_token + " " + e1
                case_sents += (" " + sep_token)
                case_sents = cls_token + " " + case_sents
            else:
                if add_sep:
                    # e1 += (" " + sep_token)
                    case_sents += (" " + sep_token)
                else:
                    # raise Exception
                    pass
                if prefix:
                    # e1 = prefix + " " + e1
                    case_sents = prefix + " " + case_sents
                    raise Exception

        # rel
        if rel_lang:
            if rel in split_into_words_atomic:
                rel = split_into_words_atomic[rel]
            else:
                print('rel:', rel)
                rel = rel
                raise Exception
            if not rel:
                print(id_line, rel, e1, e2)
                raise Exception
        else:
            rel = rel.lower()
        # JUSTTRY:
        if args.if_without_case:
            # case_sents = sep_token
            case_sents = " "
            # only print once
            if id_line == 0:
                print('INFO: No cases are allowed to be used')
        output.append((case_sents, e1, rel, e2, label, id_line))
    # print some samples for sure
    print(output[-3:])
    return output


def save_model(model, tokenizer, output_dir):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    # tokenizer.save_vocabulary(output_dir)
    tokenizer.save_pretrained(output_dir)

def add_special_tokens(args, tokenizer):
    # Q: not adding these special tokens in few shot setting (to prevent repetition problem during generation)
    if args.if_not_adding_special_relation_tokens:
        print("INFO: Not adding special relation tokens")
    else:
        if args.dataset_selection == 1:
            rel_categories = ["oReact", "oEffect", "oWant", "xAttr", "xEffect", "xIntent", \
            "xNeed", "xReact", "xWant"]
            for rel in rel_categories:
                # TOCHECK: lower()?
                tokenizer.add_tokens(['<' + rel.lower() + '>'])
        elif args.dataset_selection == 0:
            tokenizer.add_tokens(["<from_CN>", "<from_VG>", "<from_FB>"])
    # print("vocab size:", len(tokenizer))
    if args.use_special_tokens_to_split_retrieved_cases:
        tokenizer.add_tokens(["<split_cases>", "<split_source/target>"])
    # add special tokens
    # print("\nspecial tokens:", tokenizer.special_tokens_map)
    if not tokenizer.cls_token and not args.if_not_adding_special_relation_tokens:
        tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if not tokenizer.sep_token:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer

def tokenize_and_encode(obj, tokenizer, model_type=None):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        # As dpr's tokenizers are the same as BertTokenizer
        if 'bert' in model_type or 'dpr' in model_type or 'bart' in model_type:
            # bert tokenizer will automatically add [CLS] at beginning and [SEP] at end; while gpt tokneizer don't
            # we will consider [CLS] and [SEP] seperately
            return tokenizer.encode(obj)[1:-1]
        elif 'gpt2' in model_type:
            return tokenizer.encode(obj)
        elif 't5' in model_type:
            return tokenizer.encode(obj)[:-1]
        else:
            raise Exception("Not supported model_type: ", model_type)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, float):
        return None
    return list(tokenize_and_encode(o, tokenizer, model_type=model_type) for o in obj)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# For generating decoder_input_ids for bart
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


## INPUT:
# encoded_datasets_generator: (#train[(e1, r, e2, label, id), ...], #eval[], #test[])
# encoded_datasets_retriever: (#train[(e1, r, e2, label, id), ...], #eval[], #test[])
## OUTPUT:
# tensor_datasets: [#train(gene_input_ids, gene_attention_mask, gene_lm_labels, \
#                           data_idx_ids, rel_collection,
#                           retr_doc_input_ids, retr_doc_attention_mask, retr_doc_segment_ids,
#                           retr_input_ids, retr_attention_mask, retr_segment_ids), #eval(), #test()]

def preprocess_datasets_for_generator_and_retriever_and_retriever_doc_ProperEOS(args, path_tensor_datasets, \
        encoded_datasets_generator, encoded_datasets_retriever, \
        tokenizer_generator, tokenizer_retriever):
    max_e1 = args.max_e1
    max_r = args.max_r
    max_e2 = args.max_e2

    tensor_datasets = []
    input_len_gene = max_e1 + max_r + max_e2
    if args.use_obj_for_retrieval:
        input_len_retr = max_e1 + max_r + max_e2
    else:
        input_len_retr = max_e1 + max_r
    if args.use_only_sub_rel_for_retrieval:
        input_len_retr_doc = max_e1 + max_r
    else:
        input_len_retr_doc = max_e1 + max_r + max_e2

    if args.generator_model_type == "gpt2-lmhead":
        generator_pad_id = tokenizer_generator.encode(tokenizer_generator.pad_token)[0]
        generator_period_id_list = tokenizer_generator.encode(".")
        generator_split_source_target_id_list = tokenizer_generator.encode("<split_source/target>")
        generator_split_cases_id_list = tokenizer_generator.encode("<split_cases>")
        generator_semicolon_id_list = tokenizer_generator.encode(";")
        generator_eos_id_list = tokenizer_generator.encode(tokenizer_generator.eos_token)
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        generator_pad_id = tokenizer_generator.encode(tokenizer_generator.pad_token)[1]
        generator_period_id_list = tokenizer_generator.encode(".")[1:-1]
        generator_split_source_target_id_list = tokenizer_generator.encode("<split_source/target>")[1:-1]
        generator_split_cases_id_list = tokenizer_generator.encode("<split_cases>")[1:-1]
        generator_semicolon_id_list = tokenizer_generator.encode(";")[1:-1]
        generator_eos_id_list = tokenizer_generator.encode(tokenizer_generator.eos_token)[1:-1]
    elif 't5' in args.generator_model_type:
        generator_pad_id = tokenizer_generator.encode(tokenizer_generator.pad_token)[0]
        generator_period_id_list = tokenizer_generator.encode(".")[:-1]
        generator_split_source_target_id_list = tokenizer_generator.encode("<split_source/target>")[:-1]
        generator_split_cases_id_list = tokenizer_generator.encode("<split_cases>")[:-1]
        generator_semicolon_id_list = tokenizer_generator.encode(";")[:-1]
        # here tokenizer_generator.encode(tokenizer_generator.eos_token) is [1] already
        generator_eos_id_list = tokenizer_generator.encode(tokenizer_generator.eos_token)
        if len(generator_eos_id_list) == 1:
            pass
        # in case in future versions generator_eos_id_list is changed
        elif len(generator_eos_id_list) == 2:
            generator_eos_id_list = generator_eos_id_list[:-1]
        else:
            raise Exception("Illegal length of generator_eos_id_list: ", generator_eos_id_list)


    # note the mapping from rel to a unique number; used for rel_collection
    # since train/val/test set are processed together here, the mapping will be the same for each set
    path_rel_id_noter = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "rel_id_noter_{}_{}.pt".format(args.generator_model_type.split('-')[0], args.dataset_selection))
    # path_rel_id_noter can be shared
    if (args.subset_selection == -1 or (args.dataset_selection != 0 and args.dataset_selection != 1)) and not os.path.exists(path_rel_id_noter):
        rel_id_noter = {}
    else:
        if not os.path.exists(path_rel_id_noter):
            raise Exception("path_rel_id_noter missing. Try to generate it using subset_selection == -1")
        rel_id_noter = torch.load(path_rel_id_noter)
        print("rel_id_noter loaded from: ", path_rel_id_noter)
    tokenizer_gene_pad_id = generator_pad_id
    tokenizer_retr_pad_id = tokenizer_retriever.encode(tokenizer_retriever.pad_token)[1]
    assert tokenizer_retriever.encode(tokenizer_retriever.pad_token)[0] == 101
    for id_dst, gene_dataset in enumerate(encoded_datasets_generator):
        ### For Generator
        n_batch_gene = len(gene_dataset)
        ## gene_doc_input_ids
        # FQ: add '* tokenizer_gene_pad_id'
        gene_doc_input_ids = np.full((n_batch_gene, input_len_gene), fill_value=1, dtype=np.int64) * tokenizer_gene_pad_id
        # gene_cur_input_ids
        gene_cur_input_ids = np.full((n_batch_gene, input_len_gene), fill_value=1, dtype=np.int64) * tokenizer_gene_pad_id
        ## data_idx_ids
        data_idx_ids = []
        # collect rel, used for fast similarity calculation in retriever
        rel_collection = []
        print('gene_dataset[0]', gene_dataset[0])
        for i, (e1, r, e2, label, id) in enumerate(gene_dataset):
            if args.dataset_selection == 1:
                if args.use_special_tokens_to_split_retrieved_cases:
                    e1_doc = e1 + generator_period_id_list + generator_split_source_target_id_list
                    e1_cur = e1 + generator_period_id_list + generator_split_source_target_id_list
                else:
                    e1_doc = e1 + generator_period_id_list
                    e1_cur = e1 + generator_period_id_list
            else:
                if args.use_special_tokens_to_split_retrieved_cases:
                    e1_doc = e1 + generator_split_source_target_id_list
                    e1_cur = e1 + generator_split_source_target_id_list
                else:
                    e1_doc = e1
                    e1_cur = e1
            if args.use_special_tokens_to_split_retrieved_cases:
                e2_doc = e2 + generator_period_id_list + generator_split_cases_id_list
            else:
                e2_doc = e2 + generator_period_id_list + generator_semicolon_id_list
            e2_cur = e2 + generator_eos_id_list
            if i == 0:
                print("tokenizer_generator.encode(tokenizer_generator.eos_token): ", generator_eos_id_list)
            # truncate if input is too long
            if len(e1_doc) > max_e1:
                print("Warning: max_e1 is not enough: ", len(e1_doc), max_e1)
                e1_doc = e1_doc[:max_e1]
                if args.use_special_tokens_to_split_retrieved_cases:
                    e1_doc[-2] = generator_period_id_list[0]
                    e1_doc[-1] = generator_split_source_target_id_list[0]
                else:
                    e1_doc[-1] = generator_period_id_list[0]
            if len(e1_cur) > max_e1:
                print("Warning: max_e1 is not enough: ", len(e1_cur), max_e1)
                e1_cur = e1_cur[:max_e1]
                if args.use_special_tokens_to_split_retrieved_cases:
                    e1_cur[-2] = generator_period_id_list[0]
                    e1_cur[-1] = generator_split_source_target_id_list[0]
                else:
                    e1_cur[-1] = generator_period_id_list[0]
            if len(r) > max_r:
                print("Warning: max_r is not enough: ", len(r), max_r)
                r = r[:max_r]
            if len(e2_doc) > max_e2:
                print("Warning: max_e2 is not enough: ", len(e2_doc), max_e2)
                e2_doc = e2_doc[:max_e2]
                e2_doc[-2] = generator_period_id_list[0]
                if args.use_special_tokens_to_split_retrieved_cases:
                    e2_doc[-1] = generator_split_cases_id_list[0]
                else:
                    e2_doc[-1] = generator_semicolon_id_list[0]
            if len(e2_cur) > max_e2:
                print("Warning: max_e2 is not enough: ", len(e2_cur), max_e2)
                e2_cur = e2_cur[:max_e2]
                e2_cur[-1] = generator_eos_id_list[0]

            # gene_doc_input_ids
            gene_doc_input_ids[i, :len(e1_doc)] = e1_doc
            gene_doc_input_ids[i, max_e1:max_e1 + len(r)] = r
            gene_doc_input_ids[i, max_e1+max_r:max_e1+max_r+len(e2_doc)] = e2_doc
            # gene_cur_input_ids
            gene_cur_input_ids[i, :len(e1_cur)] = e1_cur
            gene_cur_input_ids[i, max_e1:max_e1 + len(r)] = r
            gene_cur_input_ids[i, max_e1+max_r:max_e1+max_r+len(e2_cur)] = e2_cur
            # data_idx_ids
            data_idx_ids.append(id)
            # get str_rel; will be used as key of dictionary
            # an example could be: '10 20 30 110'
            str_rel = [str(d) for d in r]
            str_rel = ' '.join(str_rel)
            if args.subset_selection == -1 or (args.dataset_selection != 0 and args.dataset_selection != 1):
                if not str_rel in rel_id_noter:
                    rel_id_noter[str_rel] = len(rel_id_noter)
            else:
                try:
                    assert str_rel in rel_id_noter
                except:
                    print("str_rel: ", str_rel)
                    print("rel_id_noter: ", rel_id_noter)
                    raise Exception
            rel_collection.append(rel_id_noter[str_rel])

        ## gene_doc_attention_mask
        gene_doc_attention_mask = (gene_doc_input_ids != tokenizer_gene_pad_id)
        ## gene_cur_attention_mask
        gene_cur_attention_mask = (gene_cur_input_ids != tokenizer_gene_pad_id)
        ## gene_doc_lm_labels
        gene_doc_lm_labels = np.full((n_batch_gene, input_len_gene), fill_value=1, dtype=np.int64) * -100
        ## gene_cur_lm_labels
        gene_cur_lm_labels = np.copy(gene_cur_input_ids)
        # do not calculate loss on paddings
        # GPT2 model requires here as -1
        gene_cur_lm_labels[gene_cur_lm_labels == tokenizer_gene_pad_id] = -100
        # do not calculate loss on sub/rel
        gene_cur_lm_labels[:, :max_e1 + max_r] = -100

        ### For Retriever
        retr_dataset = encoded_datasets_retriever[id_dst]
        n_batch_retr = len(retr_dataset)
        assert n_batch_retr == n_batch_gene
        ## retr_input_ids
        retr_input_ids = np.full((n_batch_retr, input_len_retr), fill_value=1, dtype=np.int64) * tokenizer_retr_pad_id
        retr_doc_input_ids = np.full((n_batch_retr, input_len_retr_doc), fill_value=1, dtype=np.int64) * tokenizer_retr_pad_id
        for i, (e1, r, e2, label, id) in enumerate(retr_dataset):
            ### model_retriever_doc
            if args.use_only_sub_rel_for_retrieval:
                e1_doc = tokenizer_retriever.encode(tokenizer_retriever.cls_token)[1:2] + e1
                r_doc = r + tokenizer_retriever.encode(tokenizer_retriever.sep_token)[1:2]
                e2_doc = None
            else:
                e1_doc = tokenizer_retriever.encode(tokenizer_retriever.cls_token)[1:2] + e1
                r_doc = r
                e2_doc = e2 + tokenizer_retriever.encode(tokenizer_retriever.sep_token)[1:2]
            ### model_retriever
            ## add [CLS] and [SEP]
            e1 = tokenizer_retriever.encode(tokenizer_retriever.cls_token)[1:2] + e1
            if args.use_obj_for_retrieval:
                r = r
                e2 = e2 + tokenizer_retriever.encode(tokenizer_retriever.sep_token)[1:2]
            else:
                r = r + tokenizer_retriever.encode(tokenizer_retriever.sep_token)[1:2]
                e2 = None
            ### model_retriever_doc
            if len(e1_doc) > max_e1:
                e1_doc = e1_doc[:max_e1]
            if len(r_doc) > max_r:
                r_doc = r_doc[:max_r]
            if not args.use_only_sub_rel_for_retrieval:
                if len(e2_doc) > max_e2:
                    e2_doc = e2_doc[:max_e2]
            ### model_retriever
            if len(e1) > max_e1:
                e1 = e1[:max_e1]
                print("Warning: max_e1 is not enough: ", len(e1), max_e1)
            if len(r) > max_r:
                r = r[:max_r]
                print("Warning: max_r is not enough: ", len(r), max_r)
            if args.use_obj_for_retrieval:
                if len(e2) > max_e2:
                    e2 = e2[:max_e2]
                    print("Warning: max_e2 is not enough: ", len(e2), max_e2)
            ### model_retriever_doc
            # retr_doc_input_ids
            retr_doc_input_ids[i, :len(e1_doc)] = e1_doc
            retr_doc_input_ids[i, max_e1:max_e1 + len(r_doc)] = r_doc
            if not args.use_only_sub_rel_for_retrieval:
                retr_doc_input_ids[i, max_e1+max_r:max_e1+max_r+len(e2_doc)] = e2_doc
            ### model_retriever
            # retr_input_ids
            retr_input_ids[i, :len(e1)] = e1
            retr_input_ids[i, max_e1:max_e1 + len(r)] = r
            if args.use_obj_for_retrieval:
                retr_input_ids[i, max_e1+max_r:max_e1+max_r+len(e2)] = e2

        ### model_retriever_doc
        retr_doc_attention_mask = (retr_doc_input_ids != tokenizer_retr_pad_id)
        retr_doc_segment_ids = np.full((n_batch_retr, input_len_retr_doc), fill_value=0, dtype=np.int64)
        ### model_retriever
        retr_attention_mask = (retr_input_ids != tokenizer_retr_pad_id)
        retr_segment_ids = np.full((n_batch_retr, input_len_retr), fill_value=0, dtype=np.int64)

        # tensor_datasets: [#train(gene_doc_input_ids, gene_doc_attention_mask, gene_doc_lm_labels,
        #                          gene_cur_input_ids, gene_cur_attention_mask, gene_cur_lm_labels, \
        #                          data_idx_ids, rel_collection, \
        #                          retr_doc_input_ids, retr_doc_attention_mask, retr_doc_segment_ids
        #                          retr_input_ids, retr_attention_mask, retr_segment_ids), #eval(), #test()]
        # print("gene_doc_lm_labels[0]: ", gene_doc_lm_labels[0])
        # print("gene_cur_lm_labels[0]: ", gene_cur_lm_labels[0])
        tensor_datasets.append((torch.tensor(gene_doc_input_ids),                               torch.tensor(gene_doc_attention_mask).to(torch.float32),
                                torch.tensor(gene_doc_lm_labels),
                                torch.tensor(gene_cur_input_ids), torch.tensor(gene_cur_attention_mask).to(torch.float32),
                                torch.tensor(gene_cur_lm_labels),
                                torch.tensor(data_idx_ids), torch.tensor(rel_collection),
                                torch.tensor(retr_doc_input_ids), torch.tensor(retr_doc_attention_mask).to(torch.float32),
                                torch.tensor(retr_doc_segment_ids),
                                torch.tensor(retr_input_ids), torch.tensor(retr_attention_mask).to(torch.float32),
                                torch.tensor(retr_segment_ids)))
    # Save for retriever
    torch.save(tensor_datasets, path_tensor_datasets)
    if (args.subset_selection == -1 and (args.dataset_selection == 0 or args.dataset_selection == 1)) and not os.path.exists(path_rel_id_noter):
        torch.save(rel_id_noter, path_rel_id_noter)
        print("rel_id_noter saved in: ", path_rel_id_noter)
    # for debug
    print(tensor_datasets[0][0][0], '\n', tensor_datasets[0][1][0], '\n', tensor_datasets[0][2][0])
    print(tensor_datasets[0][3][0], '\n', tensor_datasets[0][4][0], '\n', tensor_datasets[0][5][0])
    print(tensor_datasets[0][6][0:5], '\n', tensor_datasets[0][7][0])
    print(tensor_datasets[0][8][0], '\n', tensor_datasets[0][9][0])
    print(tensor_datasets[0][11][0], '\n', tensor_datasets[0][12][0])
    print('Successfully get and save tensor_datasets!')
    return tensor_datasets




# FUNCTION:
#   remove stop words using nltk package
# INPUT:
#   e2: a text sentence
# OUTPUT:
#   e2_NoStopWords: a text sentence
def remove_stop_words_nltk(e2):
    stop_words = set(stopwords.words('english'))
    e2_tokens = word_tokenize(e2)
    e2_tokens_NoStopWords = []
    for tmp_i, tmp_word in enumerate(e2_tokens):
        if tmp_word not in stop_words:
            e2_tokens_NoStopWords.append(tmp_word)

    e2_NoStopWords = TreebankWordDetokenizer().detokenize(e2_tokens_NoStopWords)
    return e2_NoStopWords


## Designed for models with Bart input format
# INPUT:
# retrieved_cases_cur_bundle: [encoded_cases_gene, encoded_cases_retr]
#       encoded_cases_gene: [doc_gene_input_ids, doc_gene_attention_mask, doc_gene_lm_labels]
#       encoded_cases_retr: [doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids]
#           doc_gene_input_ids: [len_bundle, n_doc, cases_per_doc * input_len_gene]
#           doc_retr_cases_input_ids: [len_bundle, n_doc, cases_per_doc, input_len_retr]
# cur_bundle: #len_bundle(gene_doc_input_ids, gene_doc_attention_mask, gene_doc_lm_labels, \
#                         gene_cur_input_ids, gene_cur_attention_mask, gene_cur_lm_labels, \
#                         data_idx_ids, rel_collection,
#                         retr_doc_input_ids, retr_doc_attention_mask, retr_doc_segment_ids,
#                         retr_input_ids, retr_attention_mask, retr_segment_ids)
# tokenizer_generator: add tokenizer.sep_token
# OUTPUT:
#   case_aug_cur_bundle: [case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels, \
# doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids, \
# input_retr_input_ids, input_retr_attention_mask, input_retr_segment_ids]
#           case_aug_gene_input_id: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           case_aug_gene_attention_mask: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           case_aug_gene_lm_labels: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           doc_retr_cases_input_ids: [len_bundle, n_doc, cases_per_doc, input_len_retr] (not changed)
#           input_retr_input_ids: [len_bundle, input_len_retr]
def concat_cur_bundle_and_encoded_cases_EOSfixed_Bart(args, cur_bundle, retrieved_cases_cur_bundle, tokenizer_generator):
    raise Exception("Not considered <s> and </s> in input and output yet")
    raise Exception("Not support GPT2 yet")
    try:
        assert cur_bundle[0].size()[0] == retrieved_cases_cur_bundle[0][0].size()[0]
    except:
        print("cur_bundle[0].size(): ", cur_bundle[0].size())
        print("retrieved_cases_cur_bundle[0][0].size()", retrieved_cases_cur_bundle[0][0].size())
        torch.save(cur_bundle, os.path.join(args.output_dir, "cur_bundle.pt"))
        torch.save(retrieved_cases_cur_bundle, os.path.join(args.output_dir, "retrieved_cases_cur_bundle.pt"))
        raise Exception("Can't agree on the same len_bundle")
    len_bundle = cur_bundle[0].size()[0]
    n_doc = retrieved_cases_cur_bundle[0][0].size()[1]
    cases_per_doc = retrieved_cases_cur_bundle[1][0].size()[2]
    input_len_gene = cur_bundle[0].size()[1]

    ## encoded_start_prompt
    if args.generator_model_type == "gpt2-lmhead":
        encoded_strt_prompt = tokenizer_generator.encode('Here are some similar cases to infer from: ')
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        encoded_strt_prompt = tokenizer_generator.encode('Here are some similar cases to infer from: ')[1:-1]
    elif "t5" in args.generator_model_type:
        encoded_strt_prompt = tokenizer_generator.encode('Here are some similar cases to infer from: ')[:-1]
    encoded_strt_prompt = torch.tensor(encoded_strt_prompt).to(torch.long).expand(len_bundle, n_doc, -1)
    attention_encoded_strt_prompt = torch.ones_like(encoded_strt_prompt).to(torch.float)
    label_encoded_strt_prompt = -100 * torch.ones_like(encoded_strt_prompt)
    ## encoded_mid_prompt
    # encoded_mid_prompt = tokenizer_generator.encode('With the similar cases we can infer that ' + tokenizer_generator.sep_token)
    if args.generator_model_type == "gpt2-lmhead":
        encoded_mid_prompt = tokenizer_generator.encode('With the similar cases we can infer that: ')
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        encoded_mid_prompt = tokenizer_generator.encode('With the similar cases we can infer that: ')[1:-1]
    elif "t5" in args.generator_model_type:
        encoded_mid_prompt = tokenizer_generator.encode('With the similar cases we can infer that: ')[:-1]
    encoded_mid_prompt = torch.tensor(encoded_mid_prompt).to(torch.long).expand(len_bundle, n_doc, -1)
    attention_encoded_mid_prompt = torch.ones_like(encoded_mid_prompt).to(torch.float)
    label_encoded_mid_prompt = -100 * torch.ones_like(encoded_mid_prompt)

    ## encoded_sep_token
    if args.generator_model_type == "gpt2-lmhead":
        encoded_sep_token = tokenizer_generator.encode(tokenizer_generator.sep_token)
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        encoded_sep_token = tokenizer_generator.encode(tokenizer_generator.sep_token)[1:-1]
    elif "t5" in args.generator_model_type:
        encoded_sep_token = tokenizer_generator.encode(tokenizer_generator.sep_token)[:-1]
    # encoded_sep_token: [len_bundle, n_doc, 1]
    encoded_sep_token = torch.tensor(encoded_sep_token).to(torch.long).expand(len_bundle, n_doc, 1)
    ## case_aug_gene_input_id
    # doc_gene_input_ids: [len_bundle, n_doc, cases_per_doc * input_len_gene]
    if args.generator_model_type == "gpt2-lmhead" or "t5" in args.generator_model_type:
        generator_pad_id = tokenizer_generator.encode(tokenizer_generator.pad_token)[0]
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        generator_pad_id = tokenizer_generator.encode(tokenizer_generator.pad_token)[1]

    if args.if_comet_baseline:
        doc_gene_input_ids = torch.ones_like(retrieved_cases_cur_bundle[0][0]) * generator_pad_id
    elif args.if_only_use_retrieved_target:
        doc_gene_input_ids = retrieved_cases_cur_bundle[0][0].view(len_bundle, n_doc, cases_per_doc, input_len_gene)
        doc_gene_input_ids = doc_gene_input_ids[:, :, :, args.max_e1+args.max_r:]
        doc_gene_input_ids = doc_gene_input_ids.reshape(len_bundle, n_doc, -1)
    elif args.if_only_use_relation_and_retrieved_target:
        doc_gene_input_ids = retrieved_cases_cur_bundle[0][0].view(len_bundle, n_doc, cases_per_doc, input_len_gene)
        doc_gene_input_ids = doc_gene_input_ids[:, :, :, args.max_e1:]
        doc_gene_input_ids = doc_gene_input_ids.reshape(len_bundle, n_doc, -1)
    else:
        doc_gene_input_ids = retrieved_cases_cur_bundle[0][0]
    # gene_cur_input_ids: [len_bundle, 1, input_len_gene]
    gene_cur_input_ids = cur_bundle[3].unsqueeze(1)
    # added for Bart
    assert gene_cur_input_ids.size()[-1] == args.max_e1 + args.max_r + args.max_e2
    gene_cur_input_ids = gene_cur_input_ids[:, :, :(args.max_e1+args.max_r)]
    # gene_cur_input_ids_e1_r = gene_cur_input_ids[:, :, :(args.max_e1+args.max_r)]
    # gene_cur_input_ids_e2 = gene_cur_input_ids[:, :, (args.max_e1+args.max_r):]
    # gene_cur_input_ids: [len_bundle, n_doc, args.max_e1+args.max_r]
    # gene_cur_input_ids_e1_r = gene_cur_input_ids_e1_r.repeat(1, n_doc, 1)
    # gene_cur_input_ids_e2 = gene_cur_input_ids_e2.repeat(1, n_doc, 1)
    gene_cur_input_ids = gene_cur_input_ids.repeat(1, n_doc, 1)
    # gene_cur_input_ids_e2 = torch.ones_like(gene_cur_input_ids_e2) * generator_pad_id
    # case_aug_gene_input_id: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
    try:
        if args.if_with_strt_mid_promp and not args.if_comet_baseline:
            # case_aug_gene_input_id: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + (args.max_e1+args.max_r) + 1]
            case_aug_gene_input_id = torch.cat((encoded_strt_prompt, doc_gene_input_ids, encoded_mid_prompt, encoded_sep_token, gene_cur_input_ids), dim=2)
        elif not args.if_comet_baseline:
            case_aug_gene_input_id = torch.cat((doc_gene_input_ids, encoded_sep_token, gene_cur_input_ids), dim=2)
        else:
            case_aug_gene_input_id = torch.cat((doc_gene_input_ids, gene_cur_input_ids), dim=2)
    except:
        print("doc_gene_input_ids.size(): ", doc_gene_input_ids.size())
        print("encoded_sep_token.size(): ", encoded_sep_token.size())
        print("gene_cur_input_ids.size()", gene_cur_input_ids.size())
        torch.save(cur_bundle, os.path.join(args.output_dir, "cur_bundle.pt"))
        torch.save(retrieved_cases_cur_bundle, os.path.join(args.output_dir, "retrieved_cases_cur_bundle.pt"))
        raise Exception("Mismatch shape between doc_gene_input_ids and gene_cur_input_ids.")
    ## case_aug_gene_attention_mask
    if args.if_comet_baseline:
        # doc_gene_attention_mask: [len_bundle, n_doc, cases_per_doc * input_len_gene]
        doc_gene_attention_mask = torch.zeros_like(retrieved_cases_cur_bundle[0][1])
    elif args.if_only_use_retrieved_target:
        doc_gene_attention_mask = retrieved_cases_cur_bundle[0][1].view(len_bundle, n_doc, cases_per_doc, input_len_gene)
        doc_gene_attention_mask = doc_gene_attention_mask[:, :, :, args.max_e1+args.max_r:]
        doc_gene_attention_mask = doc_gene_attention_mask.reshape(len_bundle, n_doc, -1)
    elif args.if_only_use_relation_and_retrieved_target:
        doc_gene_attention_mask = retrieved_cases_cur_bundle[0][1].view(len_bundle, n_doc, cases_per_doc, input_len_gene)
        doc_gene_attention_mask = doc_gene_attention_mask[:, :, :, args.max_e1:]
        doc_gene_attention_mask = doc_gene_attention_mask.reshape(len_bundle, n_doc, -1)
    else:
        doc_gene_attention_mask = retrieved_cases_cur_bundle[0][1]
    # attention_encoded_sep_token: [len_bundle, n_doc, 1]
    attention_encoded_sep_token = torch.ones((len_bundle, n_doc, 1))
    # gene_cur_attention_mask: [len_bundle, 1, input_len_gene]
    gene_cur_attention_mask = cur_bundle[4].unsqueeze(1)
    assert gene_cur_attention_mask.size()[-1] == args.max_e1 + args.max_r + args.max_e2
    gene_cur_attention_mask = gene_cur_attention_mask[:, :, :(args.max_e1+args.max_r)]
    # gene_cur_attention_mask_e1_r = gene_cur_attention_mask[:, :, :(args.max_e1+args.max_r)]
    # gene_cur_attention_mask_e2 = gene_cur_attention_mask[:, :, (args.max_e1+args.max_r):]
    # gene_cur_attention_mask: [len_bundle, n_doc, args.max_e1+args.max_r]
    # gene_cur_attention_mask_e1_r = gene_cur_attention_mask_e1_r.repeat(1, n_doc, 1)
    gene_cur_attention_mask = gene_cur_attention_mask.repeat(1, n_doc, 1)
    # gene_cur_attention_mask_e2 = gene_cur_attention_mask_e2.repeat(1, n_doc, 1)
    # gene_cur_attention_mask_e2 = torch.zeros_like(gene_cur_attention_mask_e2)
    if args.if_with_strt_mid_promp and not args.if_comet_baseline:
        case_aug_gene_attention_mask = torch.cat((attention_encoded_strt_prompt, doc_gene_attention_mask, attention_encoded_mid_prompt, attention_encoded_sep_token, gene_cur_attention_mask), dim=2)
    elif not args.if_comet_baseline:
        case_aug_gene_attention_mask = torch.cat((doc_gene_attention_mask, attention_encoded_sep_token, gene_cur_attention_mask), dim=2)
    else:
        case_aug_gene_attention_mask = torch.cat((doc_gene_attention_mask, gene_cur_attention_mask), dim=2)
    ## case_aug_gene_lm_labels
    if args.if_only_use_retrieved_target:
        doc_gene_lm_labels = retrieved_cases_cur_bundle[0][2].view(len_bundle, n_doc, cases_per_doc, input_len_gene)
        doc_gene_lm_labels = doc_gene_lm_labels[:, :, :, args.max_e1+args.max_r:]
        doc_gene_lm_labels = doc_gene_lm_labels.reshape(len_bundle, n_doc, -1)
    elif args.if_only_use_relation_and_retrieved_target:
        doc_gene_lm_labels = retrieved_cases_cur_bundle[0][2].view(len_bundle, n_doc, cases_per_doc, input_len_gene)
        doc_gene_lm_labels = doc_gene_lm_labels[:, :, :, args.max_e1:]
        doc_gene_lm_labels = doc_gene_lm_labels.reshape(len_bundle, n_doc, -1)
    else:
        # doc_gene_lm_labels: [len_bundle, n_doc, cases_per_doc * input_len_gene]
        doc_gene_lm_labels = retrieved_cases_cur_bundle[0][2]
    # label_encoded_sep_token: [len_bundle, n_doc, 1]
    label_encoded_sep_token = (-100) * torch.ones((len_bundle, n_doc, 1)).to(torch.long)
    # gene_cur_lm_labels: [len_bundle, 1, input_len_gene]
    gene_cur_lm_labels = cur_bundle[5].unsqueeze(1)
    # gene_cur_lm_labels: [len_bundle, n_doc, input_len_gene]
    assert gene_cur_lm_labels.size()[-1] == args.max_e1 + args.max_r + args.max_e2
    gene_cur_lm_labels = gene_cur_lm_labels.repeat(1, n_doc, 1)

    padding_for_lm_labels = (-100)*torch.ones((len_bundle, n_doc, case_aug_gene_input_id.size()[2]-args.max_e2)).to(torch.long)
    if args.if_with_strt_mid_promp and not args.if_comet_baseline:
        case_aug_gene_lm_labels = torch.cat((gene_cur_lm_labels[:, :, (args.max_e1+args.max_r):], padding_for_lm_labels), dim=2)
        # case_aug_gene_lm_labels = torch.cat((label_encoded_strt_prompt, doc_gene_lm_labels, label_encoded_mid_prompt, label_encoded_sep_token, gene_cur_lm_labels), dim=2)
    elif not args.if_comet_baseline:
        case_aug_gene_lm_labels = torch.cat((gene_cur_lm_labels[:, :, (args.max_e1+args.max_r):], padding_for_lm_labels), dim=2)
        # case_aug_gene_lm_labels = torch.cat((doc_gene_lm_labels, label_encoded_sep_token, gene_cur_lm_labels), dim=2)
    else:
        case_aug_gene_lm_labels = torch.cat((gene_cur_lm_labels[:, :, (args.max_e1+args.max_r):], padding_for_lm_labels), dim=2)
        # case_aug_gene_lm_labels = torch.cat((doc_gene_lm_labels, gene_cur_lm_labels), dim=2)

    assert case_aug_gene_lm_labels.size() == case_aug_gene_input_id.size()
    assert case_aug_gene_lm_labels.size() == case_aug_gene_attention_mask.size()

    ## case_aug_cur_bundle
    case_aug_cur_bundle = [case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels, \
        retrieved_cases_cur_bundle[1][0], retrieved_cases_cur_bundle[1][1], retrieved_cases_cur_bundle[1][2], \
        cur_bundle[-3], cur_bundle[-2], cur_bundle[-1]]
    return case_aug_cur_bundle






## Designed for models with Bart input format
# INPUT:
# retrieved_cases_cur_bundle: [encoded_cases_gene, encoded_cases_retr]
#       encoded_cases_gene: [doc_gene_input_ids, doc_gene_attention_mask, doc_gene_lm_labels]
#       encoded_cases_retr: [doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids]
#           doc_gene_input_ids: [len_bundle, n_doc, cases_per_doc * input_len_gene]
#           doc_retr_cases_input_ids: [len_bundle, n_doc, cases_per_doc, input_len_retr]
# cur_bundle: #len_bundle(gene_doc_input_ids, gene_doc_attention_mask, gene_doc_lm_labels, \
#                         gene_cur_input_ids, gene_cur_attention_mask, gene_cur_lm_labels, \
#                         data_idx_ids, rel_collection,
#                         retr_doc_input_ids, retr_doc_attention_mask, retr_doc_segment_ids,
#                         retr_input_ids, retr_attention_mask, retr_segment_ids)
# tokenizer_generator: add tokenizer.sep_token
# OUTPUT:
#   case_aug_cur_bundle: [case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels, \
# doc_retr_cases_input_ids, doc_retr_cases_attention_mask, doc_retr_cases_segment_ids, \
# input_retr_input_ids, input_retr_attention_mask, input_retr_segment_ids]
#           case_aug_gene_input_id: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           case_aug_gene_attention_mask: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           case_aug_gene_lm_labels: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]
#           doc_retr_cases_input_ids: [len_bundle, n_doc, cases_per_doc, input_len_retr] (not changed)
#           input_retr_input_ids: [len_bundle, input_len_retr]
def concat_cur_bundle_and_encoded_cases_EOSfixed_Bart_randomly_mask_demonstrations(args, cur_bundle, retrieved_cases_cur_bundle, tokenizer_generator, data_type):
    assert data_type == 'train' or data_type == 'eval' or data_type == 'test'
    if args.if_randomly_mask_demonstrations and data_type == 'train':
        prob_randomly_mask_demonstrations = args.prob_randomly_mask_demonstrations
    else:
        prob_randomly_mask_demonstrations = 0.0
    print("current prob_randomly_mask_demonstrations: ", prob_randomly_mask_demonstrations)
    # This function is only designed for Bart, as its output is not designed as a language modeling output; whether t5 fits for this function needs more investigation (2022.10.5); Now GPT-2 is also supported (2023.02.06)
    assert "bart" in args.generator_model_type or "gpt2" in args.generator_model_type
    if not args.if_comet_baseline:
        try:
            assert cur_bundle[0].size()[0] == retrieved_cases_cur_bundle[0][0].size()[0]
        except:
            print("cur_bundle[0].size(): ", cur_bundle[0].size())
            print("retrieved_cases_cur_bundle[0][0].size()", retrieved_cases_cur_bundle[0][0].size())
            torch.save(cur_bundle, os.path.join(args.output_dir, "cur_bundle.pt"))
            torch.save(retrieved_cases_cur_bundle, os.path.join(args.output_dir, "retrieved_cases_cur_bundle.pt"))
            raise Exception("Can't agree on the same len_bundle")
    len_bundle = cur_bundle[0].size()[0]
    # n_doc = retrieved_cases_cur_bundle[0][0].size()[1]
    # cases_per_doc = retrieved_cases_cur_bundle[1][0].size()[2]
    n_doc = args.n_doc
    cases_per_doc = args.num_cases_per_query
    if not args.if_comet_baseline:
        assert args.n_doc == retrieved_cases_cur_bundle[0][0].size()[1]
        assert args.num_cases_per_query == retrieved_cases_cur_bundle[1][0].size()[2]
    input_len_gene = cur_bundle[0].size()[1]

    ## encoded_start_prompt
    if args.generator_model_type == "gpt2-lmhead":
        encoded_strt_prompt = tokenizer_generator.encode('Here are some similar cases to infer from: ')
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        encoded_strt_prompt = tokenizer_generator.encode('Here are some similar cases to infer from: ')[1:-1]
    elif "t5" in args.generator_model_type:
        encoded_strt_prompt = tokenizer_generator.encode('Here are some similar cases to infer from: ')[:-1]
    else:
        raise NotImplementError
    encoded_strt_prompt = torch.tensor(encoded_strt_prompt).to(torch.long).expand(len_bundle, n_doc, -1)
    attention_encoded_strt_prompt = torch.ones_like(encoded_strt_prompt).to(torch.float)
    label_encoded_strt_prompt = -100 * torch.ones_like(encoded_strt_prompt)
    ## encoded_mid_prompt
    # encoded_mid_prompt = tokenizer_generator.encode('With the similar cases we can infer that ' + tokenizer_generator.sep_token)
    if args.generator_model_type == "gpt2-lmhead":
        encoded_mid_prompt = tokenizer_generator.encode('With the similar cases we can infer that: ')
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        encoded_mid_prompt = tokenizer_generator.encode('With the similar cases we can infer that: ')[1:-1]
    elif "t5" in args.generator_model_type:
        encoded_mid_prompt = tokenizer_generator.encode('With the similar cases we can infer that: ')[:-1]
    else:
        raise NotImplementError
    encoded_mid_prompt = torch.tensor(encoded_mid_prompt).to(torch.long).expand(len_bundle, n_doc, -1)
    attention_encoded_mid_prompt = torch.ones_like(encoded_mid_prompt).to(torch.float)
    label_encoded_mid_prompt = -100 * torch.ones_like(encoded_mid_prompt)

    ## encoded_sep_token --- not used in BART experiments (2023/01/31 updated), cause the sep token in BART experiment is </s>, which represents the token for ending a sentence
    # if args.generator_model_type == "gpt2-lmhead":
    #     encoded_sep_token = tokenizer_generator.encode(tokenizer_generator.sep_token)
    # elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
    #     encoded_sep_token = tokenizer_generator.encode(tokenizer_generator.sep_token)[1:-1]
    # elif "t5" in args.generator_model_type:
    #     encoded_sep_token = tokenizer_generator.encode(tokenizer_generator.sep_token)[:-1]
    # # encoded_sep_token: [len_bundle, n_doc, 1]
    # encoded_sep_token = torch.tensor(encoded_sep_token).to(torch.long).expand(len_bundle, n_doc, 1)

    # encoded_start_toekn (bart experiments, it is <s>) and encoded_end_token (bart experiments, it is </s>)
    if args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        encoded_start_token = tokenizer_generator.encode("Cheer up!")[0]
        encoded_end_token = tokenizer_generator.encode("Cheer up!")[-1]
    elif "gpt2" in args.generator_model_type:
        encoded_start_token = tokenizer_generator.encode(tokenizer_generator.bos_token)
        encoded_end_token = tokenizer_generator.encode(tokenizer_generator.eos_token)
    else:
        raise NotImplementError
    encoded_start_token = torch.tensor(encoded_start_token).to(torch.long).expand(len_bundle, n_doc, 1)
    encoded_end_token = torch.tensor(encoded_end_token).to(torch.long).expand(len_bundle, n_doc, 1)


    ## case_aug_gene_input_id
    # doc_gene_input_ids: [len_bundle, n_doc, cases_per_doc * input_len_gene]
    if args.generator_model_type == "gpt2-lmhead" or "t5" in args.generator_model_type:
        generator_pad_id = tokenizer_generator.encode(tokenizer_generator.pad_token)[0]
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        generator_pad_id = tokenizer_generator.encode(tokenizer_generator.pad_token)[1]
    else:
        raise NotImplementError

    if args.if_comet_baseline:
        if retrieved_cases_cur_bundle != None:
            doc_gene_input_ids = torch.ones_like(retrieved_cases_cur_bundle[0][0]) * generator_pad_id
        else:
            doc_gene_input_ids = torch.ones((len_bundle, n_doc, cases_per_doc * input_len_gene), dtype=torch.long) * generator_pad_id
    elif args.if_only_use_retrieved_target:
        doc_gene_input_ids = retrieved_cases_cur_bundle[0][0].view(len_bundle, n_doc, cases_per_doc, input_len_gene)
        doc_gene_input_ids = doc_gene_input_ids[:, :, :, args.max_e1+args.max_r:]
        doc_gene_input_ids = doc_gene_input_ids.reshape(len_bundle, n_doc, -1)
    elif args.if_only_use_relation_and_retrieved_target:
        doc_gene_input_ids = retrieved_cases_cur_bundle[0][0].view(len_bundle, n_doc, cases_per_doc, input_len_gene)
        doc_gene_input_ids = doc_gene_input_ids[:, :, :, args.max_e1:]
        doc_gene_input_ids = doc_gene_input_ids.reshape(len_bundle, n_doc, -1)
    else:
        doc_gene_input_ids = retrieved_cases_cur_bundle[0][0]
    # gene_cur_input_ids: [len_bundle, 1, input_len_gene]
    gene_cur_input_ids = cur_bundle[3].unsqueeze(1)
    # added for Bart
    assert gene_cur_input_ids.size()[-1] == args.max_e1 + args.max_r + args.max_e2

    # gene_cur_input_ids
    if "bart" in args.generator_model_type:
        gene_cur_input_ids = gene_cur_input_ids[:, :, :(args.max_e1+args.max_r)]
    elif "gpt2" in args.generator_model_type:
        pass
    else:
        raise NotImplementError
    # gene_cur_input_ids_e1_r = gene_cur_input_ids[:, :, :(args.max_e1+args.max_r)]
    # gene_cur_input_ids_e2 = gene_cur_input_ids[:, :, (args.max_e1+args.max_r):]
    # gene_cur_input_ids: [len_bundle, n_doc, args.max_e1+args.max_r]
    # gene_cur_input_ids_e1_r = gene_cur_input_ids_e1_r.repeat(1, n_doc, 1)
    # gene_cur_input_ids_e2 = gene_cur_input_ids_e2.repeat(1, n_doc, 1)
    gene_cur_input_ids = gene_cur_input_ids.repeat(1, n_doc, 1)
    # gene_cur_input_ids_e2 = torch.ones_like(gene_cur_input_ids_e2) * generator_pad_id
    # case_aug_gene_input_id: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + input_len_gene]

    # print("gene_cur_input_ids: ", gene_cur_input_ids)
    if args.if_with_strt_mid_promp and not args.if_comet_baseline:
        if "bart" in args.generator_model_type:
            # case_aug_gene_input_id: [len_bundle, n_doc, cases_per_doc * input_len_gene + 1 + (args.max_e1+args.max_r) + 1]
            # case_aug_gene_input_id = torch.cat((encoded_strt_prompt, doc_gene_input_ids, encoded_mid_prompt, encoded_sep_token, gene_cur_input_ids), dim=2)
            # not using <sep> token in bart experiments, but newly using start and end tokens in bart experiments
            # case_aug_gene_input_id: [len_bundle, n_doc, cases_per_doc * input_len_gene + (args.max_e1+args.max_r) + 1 + 2]
            case_aug_gene_input_id = torch.cat((encoded_start_token, encoded_strt_prompt, doc_gene_input_ids, encoded_mid_prompt, gene_cur_input_ids, encoded_end_token), dim=2)
        elif "gpt2" in args.generator_model_type:
            case_aug_gene_input_id = torch.cat((encoded_start_token, encoded_strt_prompt, doc_gene_input_ids, encoded_mid_prompt, gene_cur_input_ids), dim=2)
        else:
            raise NotImplementError
    # elif not args.if_comet_baseline:
    # no matter args.if_comet_baseline, here should be the same
    else:
        if "bart" in args.generator_model_type:
            # case_aug_gene_input_id = torch.cat((doc_gene_input_ids, encoded_sep_token, gene_cur_input_ids), dim=2)
            # not using <sep> token in bart experiments
            case_aug_gene_input_id = torch.cat((encoded_start_token, doc_gene_input_ids, gene_cur_input_ids, encoded_end_token), dim=2)
        elif "gpt2" in args.generator_model_type:
            case_aug_gene_input_id = torch.cat((encoded_start_token, doc_gene_input_ids, gene_cur_input_ids), dim=2)
        else:
            raise NotImplementError
    # else:
    #     raise NotImplementError("comet baseline is implemented in another function")

    # case_aug_gene_input_id_noRetrieval
    if "bart" in args.generator_model_type:
        # (- 2 because to leave the space for start token and end token)
        case_aug_gene_input_id_noRetrieval_padding = torch.ones(case_aug_gene_input_id.size()[0], case_aug_gene_input_id.size()[1], case_aug_gene_input_id.size()[2] - gene_cur_input_ids.size()[2] - 2) * generator_pad_id
        case_aug_gene_input_id_noRetrieval = torch.cat((encoded_start_token, case_aug_gene_input_id_noRetrieval_padding, gene_cur_input_ids, encoded_end_token), dim=2)
    elif "gpt2" in args.generator_model_type:
        # (- 1 because to leave the space for start token)
        case_aug_gene_input_id_noRetrieval_padding = torch.ones(case_aug_gene_input_id.size()[0], case_aug_gene_input_id.size()[1], case_aug_gene_input_id.size()[2] - gene_cur_input_ids.size()[2] - 1) * generator_pad_id
        case_aug_gene_input_id_noRetrieval = torch.cat((encoded_start_token, case_aug_gene_input_id_noRetrieval_padding, gene_cur_input_ids), dim=2)
    else:
        raise NotImplementError

    ## case_aug_gene_lm_labels
    # if args.if_only_use_retrieved_target:
    #     doc_gene_lm_labels = retrieved_cases_cur_bundle[0][2].view(len_bundle, n_doc, cases_per_doc, input_len_gene)
    #     doc_gene_lm_labels = doc_gene_lm_labels[:, :, :, args.max_e1+args.max_r:]
    #     doc_gene_lm_labels = doc_gene_lm_labels.reshape(len_bundle, n_doc, -1)
    # elif args.if_only_use_relation_and_retrieved_target:
    #     doc_gene_lm_labels = retrieved_cases_cur_bundle[0][2].view(len_bundle, n_doc, cases_per_doc, input_len_gene)
    #     doc_gene_lm_labels = doc_gene_lm_labels[:, :, :, args.max_e1:]
    #     doc_gene_lm_labels = doc_gene_lm_labels.reshape(len_bundle, n_doc, -1)
    # elif not args.if_comet_baseline:
    #     # doc_gene_lm_labels: [len_bundle, n_doc, cases_per_doc * input_len_gene]
    #     doc_gene_lm_labels = retrieved_cases_cur_bundle[0][2]
    # else:
    #     raise NotImplementError("comet baseline is implemented in another function")

    # # label_encoded_sep_token: [len_bundle, n_doc, 1]
    # label_encoded_sep_token = (-100) * torch.ones((len_bundle, n_doc, 1)).to(torch.long)
    # gene_cur_lm_labels: [len_bundle, 1, input_len_gene]
    gene_cur_lm_labels = cur_bundle[5].unsqueeze(1)
    # gene_cur_lm_labels: [len_bundle, n_doc, input_len_gene]
    assert gene_cur_lm_labels.size()[-1] == args.max_e1 + args.max_r + args.max_e2
    gene_cur_lm_labels = gene_cur_lm_labels.repeat(1, n_doc, 1)

    if "bart" in args.generator_model_type:
        # -1 to leave space for start tokens (<s>); end token has already been accidentally added to gene_cur_lm_labels
        padding_for_lm_labels = (-100)*torch.ones((len_bundle, n_doc, case_aug_gene_input_id.size()[2]-args.max_e2-1)).to(torch.long)
        if args.if_with_strt_mid_promp and not args.if_comet_baseline:
            case_aug_gene_lm_labels = torch.cat((encoded_start_token, gene_cur_lm_labels[:, :, (args.max_e1+args.max_r):], padding_for_lm_labels), dim=2)
            # case_aug_gene_lm_labels = torch.cat((label_encoded_strt_prompt, doc_gene_lm_labels, label_encoded_mid_prompt, label_encoded_sep_token, gene_cur_lm_labels), dim=2)
        # elif not args.if_comet_baseline:
        # no matter args.if_comet_baseline, here should be the same
        else:
            case_aug_gene_lm_labels = torch.cat((encoded_start_token, gene_cur_lm_labels[:, :, (args.max_e1+args.max_r):], padding_for_lm_labels), dim=2)
            # case_aug_gene_lm_labels = torch.cat((doc_gene_lm_labels, label_encoded_sep_token, gene_cur_lm_labels), dim=2)
        # else:
        #     raise NotImplementError("comet baseline is implemented in another function")
        # else:
            # case_aug_gene_lm_labels = torch.cat((gene_cur_lm_labels[:, :, (args.max_e1+args.max_r):], padding_for_lm_labels), dim=2)
            # # case_aug_gene_lm_labels = torch.cat((doc_gene_lm_labels, gene_cur_lm_labels), dim=2)
    elif "gpt2" in args.generator_model_type:
        # case_aug_gene_input_id = torch.cat((encoded_start_token, doc_gene_input_ids, gene_cur_input_ids), dim=2)
        # gene_cur_lm_labels
        case_aug_gene_lm_labels = (-100) * np.full((case_aug_gene_input_id.size()[0], case_aug_gene_input_id.size()[1], case_aug_gene_input_id.size()[2]), fill_value=1, dtype=np.int64)
        case_aug_gene_lm_labels = np.copy(case_aug_gene_input_id)
        case_aug_gene_lm_labels[case_aug_gene_input_id == generator_pad_id] = -100
        case_aug_gene_lm_labels[:, :, :-args.max_e2] = -100
        case_aug_gene_lm_labels = torch.from_numpy(case_aug_gene_lm_labels)
    else:
        raise NotImplementError


    # case_aug_gene_lm_labels_noRetrieval is the same as case_aug_gene_lm_labels since case_aug_gene_input_id_noRetrieval pads in the front
    assert case_aug_gene_lm_labels.size() == case_aug_gene_input_id.size()
    # assert case_aug_gene_lm_labels.size() == case_aug_gene_attention_mask.size()
    assert case_aug_gene_input_id.size() == case_aug_gene_input_id_noRetrieval.size()
    # assert case_aug_gene_attention_mask.size() == case_aug_gene_attention_mask_noRetrieval.size()

    ## combine case_aug_gene_input_id and case_aug_gene_input_id_noRetrieval
    # random selection
    full_ids = np.arange(case_aug_gene_input_id.size()[0], dtype=np.longlong).tolist()
    ids_withRetrieval, ids_withoutRetrieval = [], []
    for id in full_ids:
        tmp_rand = np.random.rand(1)[0]
        if tmp_rand >= prob_randomly_mask_demonstrations:
            ids_withRetrieval.append(id)
        else:
            ids_withoutRetrieval.append(id)
    ids_withRetrieval = torch.tensor(ids_withRetrieval)
    ids_withoutRetrieval = torch.tensor(ids_withoutRetrieval)
    # print("ids_withRetrieval: ", ids_withRetrieval)
    # print("ids_withoutRetrieval: ", ids_withoutRetrieval)
    # final_case_aug_gene_input_id, final_case_aug_gene_attention_mask
    for id in range(len(full_ids)):
        if id in ids_withRetrieval:
            tmp_case_aug_gene_input_id = case_aug_gene_input_id[id, :, :].unsqueeze(0)
            # tmp_case_aug_gene_attention_mask = case_aug_gene_attention_mask[id, :, :].unsqueeze(0)
        elif id in ids_withoutRetrieval:
            tmp_case_aug_gene_input_id = case_aug_gene_input_id_noRetrieval[id, :, :].unsqueeze(0)
            # tmp_case_aug_gene_attention_mask = case_aug_gene_attention_mask_noRetrieval[id, :, :].unsqueeze(0)
        else:
            raise Exception("Wrong value for id: ", id)
        # concat
        if id == 0:
            final_case_aug_gene_input_id = tmp_case_aug_gene_input_id
            # final_case_aug_gene_attention_mask = tmp_case_aug_gene_attention_mask
        else:
            final_case_aug_gene_input_id = torch.cat((final_case_aug_gene_input_id, tmp_case_aug_gene_input_id), dim=0)
            # final_case_aug_gene_attention_mask = torch.cat((final_case_aug_gene_attention_mask, tmp_case_aug_gene_attention_mask), dim=0)

    # final_case_aug_gene_attention_mask
    if "bart" in args.generator_model_type:
        # final_case_aug_gene_attention_mask
        final_case_aug_gene_attention_mask = (final_case_aug_gene_input_id != generator_pad_id)
    elif "gpt2" in args.generator_model_type:
        # final_case_aug_gene_attention_mask
        final_case_aug_gene_attention_mask = (final_case_aug_gene_input_id != generator_pad_id)
        final_case_aug_gene_attention_mask[:, :, -args.max_e2:] = 0
        # final_case_aug_gene_input_id
        final_case_aug_gene_input_id[:, :, -args.max_e2:] =  generator_pad_id
    else:
        raise NotImplementError
    # final_case_aug_gene_lm_labels
    final_case_aug_gene_lm_labels = case_aug_gene_lm_labels

    assert final_case_aug_gene_input_id.size() == final_case_aug_gene_attention_mask.size()
    assert final_case_aug_gene_input_id.size() == final_case_aug_gene_lm_labels.size()

    ## case_aug_cur_bundle
    # case_aug_cur_bundle = [case_aug_gene_input_id, case_aug_gene_attention_mask, case_aug_gene_lm_labels, \
    #     retrieved_cases_cur_bundle[1][0], retrieved_cases_cur_bundle[1][1], retrieved_cases_cur_bundle[1][2], \
    #     cur_bundle[-3], cur_bundle[-2], cur_bundle[-1]]

    # More than 3 return values are only useful for --if_eval_analysis (we dont use --if_eval_analysis now at all)
    # case_aug_cur_bundle = [final_case_aug_gene_input_id.to(torch.long), final_case_aug_gene_attention_mask.to(torch.long), final_case_aug_gene_lm_labels, \
    #     retrieved_cases_cur_bundle[1][0], retrieved_cases_cur_bundle[1][1], retrieved_cases_cur_bundle[1][2], \
    #     cur_bundle[-3], cur_bundle[-2], cur_bundle[-1]]
    case_aug_cur_bundle = [final_case_aug_gene_input_id.to(torch.long), final_case_aug_gene_attention_mask.to(torch.long), final_case_aug_gene_lm_labels]
    return case_aug_cur_bundle



## OUTPUT:
# path_prev_next_bundle
# path_next_bundle
def get_path_cur_next_bundle(args, path_cnt_saved_bundle, data_type, path_next_bundle_train=None, path_next_bundle_eval=None, path_next_bundle_test=None):
    if data_type == 'train':
        path_next_bundle = path_next_bundle_train
        id_cnt_saved_bundle = 0
    elif data_type == 'eval':
        path_next_bundle = path_next_bundle_eval
        id_cnt_saved_bundle = 1
    elif data_type == 'test':
        path_next_bundle = path_next_bundle_test
        id_cnt_saved_bundle = 2
    else:
        raise Exception
    assert path_next_bundle != None

    if not os.path.exists(path_cnt_saved_bundle):
        cnt_saved_bundle = [0, 0, 0]
        torch.save(cnt_saved_bundle, path_cnt_saved_bundle)
    else:
        cnt_saved_bundle = torch.load(path_cnt_saved_bundle)

    path_next_bundle = path_next_bundle.split('.')
    # path_prev_next_bundle
    path_prev_next_bundle = copy.copy(path_next_bundle)
    path_prev_next_bundle[0] = path_prev_next_bundle[0] + '_' + str(cnt_saved_bundle[id_cnt_saved_bundle])
    path_prev_next_bundle = '.'.join(path_prev_next_bundle)
    # path_next_bundle
    path_next_bundle[0] = path_next_bundle[0] + '_' + str(cnt_saved_bundle[id_cnt_saved_bundle] + 1)
    path_next_bundle = '.'.join(path_next_bundle)

    if cnt_saved_bundle[id_cnt_saved_bundle] <= 3:
        print("path_prev_next_bundle: ", path_prev_next_bundle)
        print("path_next_bundle: ", path_next_bundle)
    return path_prev_next_bundle, path_next_bundle, cnt_saved_bundle


# interact with retriever, get retrieved_cases for cur_batches
# retrieved_cases_cur_batches: in txt format; [num_batch, batch, n_doc]
def wait_get_remove_cases_for_bundle_while_deleting_bad_cases_file(args, path_cnt_retrieved_bundle, data_type, path_retrieved_encoded_cases_train=None, path_retrieved_encoded_cases_eval=None, path_retrieved_encoded_cases_test=None):
    ## get path_path_retrieved_encoded_cases
    if data_type == 'train':
        path_retrieved_encoded_cases = path_retrieved_encoded_cases_train
        id_cnt_retrieved_bundle = 0
        encoded_cases_prompt = 'encoded_cases_train'
    elif data_type == 'eval':
        path_retrieved_encoded_cases = path_retrieved_encoded_cases_eval
        id_cnt_retrieved_bundle = 1
        encoded_cases_prompt = 'encoded_cases_eval'
    elif data_type == 'test':
        path_retrieved_encoded_cases = path_retrieved_encoded_cases_test
        id_cnt_retrieved_bundle = 2
        encoded_cases_prompt = 'encoded_cases_test'
    else:
        raise Exception
    assert path_retrieved_encoded_cases != None

    if not os.path.exists(path_cnt_retrieved_bundle):
        cnt_retrieved_bundle = [0,0,0]
        torch.save(cnt_retrieved_bundle, path_cnt_retrieved_bundle)
    else:
        cnt_retrieved_bundle = torch.load(path_cnt_retrieved_bundle)

    path_retrieved_encoded_cases = path_retrieved_encoded_cases.split('.')
    path_retrieved_encoded_cases[0] += '_' + str(cnt_retrieved_bundle[id_cnt_retrieved_bundle]+1)
    path_retrieved_encoded_cases = '.'.join(path_retrieved_encoded_cases)

    print('Waiting for encoded_cases...')
    start_waiting_time = time.time()
    while(not os.path.exists(path_retrieved_encoded_cases)):
        time.sleep(5)
        possible_other_encoded_cases_files = [os.path.join(args.output_dir, i) for i in os.listdir(args.output_dir) if i.startswith(encoded_cases_prompt)]
        if len(possible_other_encoded_cases_files) > 1 or \
            (len(possible_other_encoded_cases_files) == 1 and path_retrieved_encoded_cases not in possible_other_encoded_cases_files):
            try:
                possible_other_encoded_cases_files.remove(path_retrieved_encoded_cases)
            except:
                print('possible_other_encoded_cases_files: ', possible_other_encoded_cases_files)
                print('path_retrieved_encoded_cases: ', path_retrieved_encoded_cases)
                possible_other_encoded_cases_files.remove(path_retrieved_encoded_cases)
            for tmp_file in possible_other_encoded_cases_files:
                print("Warning: {} exists while the model is waiting for {}".format(tmp_file, path_retrieved_encoded_cases))
                os.remove(tmp_file)

    time.sleep(3)
    print('--- Wait for cases: %s seconds ---' % (time.time() - start_waiting_time))
    while True:
        try:
            retrieved_cases_cur_batches = torch.load(path_retrieved_encoded_cases)
            break
        except:
            time.sleep(5)
    os.remove(path_retrieved_encoded_cases)
    assert not os.path.exists(path_retrieved_encoded_cases)
    # update cnt_retrieved_bundle
    cnt_retrieved_bundle[id_cnt_retrieved_bundle] += 1
    torch.save(cnt_retrieved_bundle, path_cnt_retrieved_bundle)
    return retrieved_cases_cur_batches

# # For gpt2; can use pre_process_datasets_withCases_Bart_or_GPT2 instead (pre_process_datasets_withCases_Bart_or_GPT2 supports both BART and GPT-2 models)
# def pre_process_datasets_withCases_GPT2(encoded_datasets, max_e1, max_r, max_e2, max_additional_cases, predict_part="obj", model_type=None, encoded_pad_token=None):
#     tensor_datasets = []
#     input_len = max_additional_cases + max_e1 + max_r
#     mask_value_on_lm_models = -100
#     for dataset in encoded_datasets:
#         n_batch = len(dataset)
#         print('n_batch: ', n_batch)
#         ## input_ids
#         input_ids = encoded_pad_token * np.full((n_batch, input_len), fill_value=1, dtype=np.int64)
#         lm_labels = mask_value_on_lm_models * np.full((n_batch, input_len), fill_value=1, dtype=np.int64)
#         ## segment_ids
#         # for bert
#         segment_ids_1 = np.full((n_batch, max_additional_cases), fill_value=0, dtype=np.int64)
#         segment_ids_2 = np.full((n_batch, max_e1 + max_e2 + max_r), fill_value=1, dtype=np.int64)
#         segment_ids = np.concatenate((segment_ids_1, segment_ids_2), 1)
#
#         data_idx_ids = []
#         for i, (cases, e1, r, e2, label, id), in enumerate(dataset):
#             # data_idx_ids
#             data_idx_ids.append(id)
#             # truncate if input is too long
#             if len(cases) > max_additional_cases:
#                 cases = cases[:max_additional_cases]
#                 print('info: max_additional_cases is not enough', len(cases))
#                 raise Exception("args.max_additional_cases: {} is not enough for len(cases): {}".format(max_additional_cases, len(cases)))
#             if len(e1) > max_e1:
#                 e1 = e1[:max_e1]
#             if len(r) > max_r:
#                 r = r[:max_r]
#             if len(e2) > max_e2:
#                 e2 = e2[:max_e2]
#
#             input_ids[i, :len(cases)] = cases
#             input_ids[i, max_additional_cases:max_additional_cases + len(e1)] = e1
#             input_ids[i, max_additional_cases+max_e1:max_additional_cases+max_e1 + len(r)] = r
#             # input_ids[i, max_additional_cases+max_e1+max_r:max_additional_cases+max_e1+max_r+len(e2)] = e2
#             if i == 0 or i == 1:
#                 print("one encoded sample: cases", cases, "e1", e1, "r", r, "e2", e2, "input_ids:", input_ids[i])
#
#         ## lm_labels
#         lm_labels[i, :len(e2)] = e2
#         lm_labels[i, len(e2):] = mask_value_on_lm_models
#         ## input_mask
#         input_mask = (input_ids != encoded_pad_token)   # attention mask
#         # all_inputs = (input_ids, lm_labels, input_mask, segment_ids)
#         tensor_datasets.append((torch.tensor(input_ids), torch.tensor(lm_labels), \
#                 torch.tensor(input_mask).to(torch.float32), torch.tensor(segment_ids), torch.tensor(data_idx_ids)))
#     return tensor_datasets


# For Bart or possibly T5
def pre_process_datasets_withCases_Bart_or_GPT2(encoded_datasets, max_e1, max_r, max_e2, max_additional_cases, tokenizer=None, predict_part="obj", model_type=None, encoded_pad_token=None):
    assert "gpt2" in model_type or "bart" in model_type
    # if not "bart" in model_type:
    #     raise Exception("Only adapted to BART model for now (<s> and </s> tokens)")
    if "bart" in model_type or 'bert' in model_type:
        encoded_start_token = tokenizer.encode("Cheer up!")[0]
        encoded_end_token = tokenizer.encode("Cheer up!")[-1]
    elif 'gpt' in model_type:
        encoded_start_token = tokenizer.encode(tokenizer.bos_token)[0]
        encoded_end_token = tokenizer.encode(tokenizer.eos_token)[0]
    else:
        raise NotImplementError

    tensor_datasets = []
    input_len = max_additional_cases + max_e1 + max_r
    # Both Bert and GPT2 require here as -100
    mask_value_on_lm_models = -100
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        print('n_batch: ', n_batch)
        ## input_ids
        input_ids = encoded_pad_token * np.full((n_batch, input_len), fill_value=1, dtype=np.int64)
        lm_labels = mask_value_on_lm_models * np.full((n_batch, input_len), fill_value=1, dtype=np.int64)
        ## segment_ids
        # for bert
        segment_ids_1 = np.full((n_batch, max_additional_cases), fill_value=0, dtype=np.int64)
        segment_ids_2 = np.full((n_batch, max_e1 + max_e2 + max_r), fill_value=1, dtype=np.int64)
        segment_ids = np.concatenate((segment_ids_1, segment_ids_2), 1)

        data_idx_ids = []
        for i, (cases, e1, r, e2, label, id), in enumerate(dataset):
            # data_idx_ids
            data_idx_ids.append(id)
            # truncate if input is too long
            if len(cases) > max_additional_cases:
                cases = cases[:max_additional_cases]
                print('info: max_additional_cases is not enough', len(cases))
                raise Exception("args.max_additional_cases: {} is not enough for len(cases): {}".format(max_additional_cases, len(cases)))
            if len(e1) > max_e1:
                e1 = e1[:max_e1]
            if len(r) > max_r:
                r = r[:max_r]
            if len(e2) > max_e2:
                e2 = e2[:max_e2]

            if 'gpt' in model_type:
                input_ids[i, 0] = encoded_start_token
                input_ids[i, 1:len(cases)+1] = cases
                input_ids[i, max_additional_cases:max_additional_cases + len(e1)] = e1
                input_ids[i, max_additional_cases+max_e1:max_additional_cases+max_e1 + len(r)] = r
                # lm_labels
                lm_labels[i, :len(e2)] = e2
                lm_labels[i, len(e2):] = mask_value_on_lm_models
            elif 'bart' in model_type or 'bert' in model_type:
                input_ids[i, 0] = encoded_start_token
                input_ids[i, 1:len(cases)+1] = cases
                input_ids[i, max_additional_cases:max_additional_cases + len(e1)] = e1
                input_ids[i, max_additional_cases+max_e1:max_additional_cases+max_e1 + len(r)] = r
                input_ids[i, max_additional_cases+max_e1+max_r:max_additional_cases+max_e1+max_r + 1] = encoded_end_token
                # lm_labels (encoded_end_token should have been added previously)
                # lm_labels[i, :len(e2)+2] = [encoded_start_token] + e2 + [encoded_end_token]
                lm_labels[i, :len(e2)+1] = [encoded_start_token] + e2
                lm_labels[i, len(e2)+1:] = mask_value_on_lm_models
            else:
                raise NotImplementError
            if i == 0 or i == 1:
                print("one encoded sample: cases", cases, "e1", e1, "r", r, "e2", e2)
                print("input_ids:", input_ids[i])
                print("lm_labels:", lm_labels[i])

        ## input_mask
        input_mask = (input_ids != encoded_pad_token)   # attention mask
        # all_inputs = (input_ids, lm_labels, input_mask, segment_ids)
        tensor_datasets.append((torch.tensor(input_ids), torch.tensor(lm_labels), \
                torch.tensor(input_mask).to(torch.float32), torch.tensor(segment_ids), torch.tensor(data_idx_ids)))
    return tensor_datasets

## INPUT
# len_similarity: length of similarity score matrix
# num_cases: number of in-context demonstrations
# filter_ratio: initial ratio to decrease len_similarity
# ini_times_of_num_cases_to_keep: initial minimal times of num_cases to keep (ini_times_of_num_cases_to_keep * num_cases)
## FUNCTION: get the k for torch.topk(similarity_subRel, k) in v6.py;
# k should be around 50% of original length, but should not be lower than 4 * num_cases
def get_k_for_topk_subRel(len_similarity, num_cases, filter_ratio=0.5, ini_times_of_num_cases_to_keep=4):
    if_decrease = True
    while len_similarity * (1-filter_ratio) <= ini_times_of_num_cases_to_keep * num_cases:
        filter_ratio *= 0.5
        # len_similarity is too less that we just keep all cases and do not decrease it
        if filter_ratio <= 0.03:
            if_decrease = False
            break
    # print("len_similarity: {}, filter_ratio: {}".format(len_similarity, filter_ratio))
    if if_decrease:
        return math.ceil(len_similarity * (1-filter_ratio))
    else:
        return len_similarity

# Get selected_idxs and selected_idx_prob from rlt_topk and rlt_subRel_topk
def get_selected_idxs_and_prob_from_double_topk_result(rlt_topk, rlt_subRel_topk):
    # rlt_topk
    selected_idxs = rlt_topk[1]
    selected_idx_prob = rlt_topk[0]
    # rlt_subRel_topk
    selected_idxs_subRel = rlt_subRel_topk[1]
    print("selected_idxs.size(): ", selected_idxs.size())
    print("selected_idxs_subRel.size(): ", selected_idxs_subRel.size())
    # print("selected_idxs[:5, :50]: ", selected_idxs[:5, :50])
    # print("selected_idxs_subRel[:5, :50]: ", selected_idxs_subRel[:5, :50])

    list_accepted_selected_idxs, list_accepted_selected_idxs_prob = [], []
    len_list_noter = []
    for tmp_row in range(selected_idxs.size()[0]):
        list_accepted_selected_idxs_tmp_row, list_accepted_selected_idxs_prob_tmp_row = [], []
        tmp_abandon_list = []
        for tmp_col in range(selected_idxs.size()[1]):
            if selected_idxs[tmp_row, tmp_col] in selected_idxs_subRel[tmp_row]:
                list_accepted_selected_idxs_tmp_row.append(selected_idxs[tmp_row, tmp_col].item())
                list_accepted_selected_idxs_prob_tmp_row.append(selected_idx_prob[tmp_row, tmp_col].item())
            else:
                tmp_abandon_list.append(tmp_col)
        list_accepted_selected_idxs.append(list_accepted_selected_idxs_tmp_row)
        list_accepted_selected_idxs_prob.append(list_accepted_selected_idxs_prob_tmp_row)
        if tmp_row == 0:
            print("tmp_abandon_list[:10]: ", tmp_abandon_list[:10])
        assert len(list_accepted_selected_idxs_tmp_row) == len(list_accepted_selected_idxs_prob_tmp_row)
        len_list_noter.append(len(list_accepted_selected_idxs_tmp_row))
    min_len_list, max_len_list = min(len_list_noter), max(len_list_noter)
    # We don't want min_len_list and max_len_list be too different
    if min_len_list < 0.5 * max_len_list:
        raise Exception("min_len_list: {}; max_len_list: {}".format(min_len_list, max_len_list))
    print("min_len_list: {}, max_len_list: {}, ori_len: {}, reference_len: {}".format(min_len_list, max_len_list, selected_idxs.size()[1], selected_idxs_subRel.size()[1]))
    # reduce len_list to min_len_list
    for id in range(len(list_accepted_selected_idxs)):
        list_accepted_selected_idxs[id] = list_accepted_selected_idxs[id][:min_len_list]
        list_accepted_selected_idxs_prob[id] = list_accepted_selected_idxs_prob[id][:min_len_list]
    # turn data type to tensor
    final_selected_idxs = torch.tensor(list_accepted_selected_idxs).to(torch.long)
    final_selected_idx_prob = torch.tensor(list_accepted_selected_idxs_prob)
    return final_selected_idxs, final_selected_idx_prob


# when larger_range_to_select_retrieval_randomly_from != -1, this function randomly selects
#       num_cases cases from larger_range_to_select_retrieval_randomly_from cases
#       (designed to be a 'dropout' style trick)
## INPUT:
# id_with_different_source: a list, with length smaller than larger_range_to_select_retrieval_randomly_from
# larger_range_to_select_retrieval_randomly_from: an int
# num_cases: an int, number of demonstrations to put in input_ids
## OUTPUT:
# id_for_id_with_different_source: a list, with length equals to num_cases
def get_id_for_id_with_different_source_during_larger_range(id_with_different_source, \
                larger_range_to_select_retrieval_randomly_from, num_cases):
    assert len(id_with_different_source) <= larger_range_to_select_retrieval_randomly_from
    assert larger_range_to_select_retrieval_randomly_from > num_cases
    id_for_id_with_different_source = []
    tmp_data_seq = np.arange(len(id_with_different_source))
    random.shuffle(tmp_data_seq)
    tmp_data_seq = tmp_data_seq[:num_cases]
    tmp_data_seq.sort()
    return tmp_data_seq


def find_path_tensor_dataset(args):
    if args.dataset_selection == 0:
        if args.use_obj_for_retrieval:
            print("INFO: using obj for retrieval")
            path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_conceptnet_using_obj_for_retrieval_twoEmbedder_usingPad_EOSadded_DEBUG_EOSdouble_END_FOLLOWED.pt")
            raise NotImplementedError
        elif args.use_only_sub_rel_for_retrieval:
            print("INFO: only using sub and rel for retrieval (to get embedding)")
            if args.use_special_tokens_to_split_retrieved_cases:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_conceptnet_use_only_sub_rel_for_retrieval_specialTokenSplit_twoEmbedder_usingPad_EOSadded_DEBUG_EOSFOLLOWED")
            else:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_conceptnet_use_only_sub_rel_for_retrieval_twoEmbedder_usingPad_EOSadded_DEBUG_EOSFOLLOWED")
        else:
            if args.use_special_tokens_to_split_retrieved_cases:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_conceptnet_specialTokenSplit_twoEmbedder_usingPad_EOSadded_DEBUG_EOSFOLLOWED.pt")
            else:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_conceptnet_twoEmbedder_usingPad_EOSadded_DEBUG_EOSFOLLOWED.pt")
    elif args.dataset_selection == 1:
        if args.use_obj_for_retrieval:
            print("INFO: using obj during retrieval")
            path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_atomic_using_obj_for_retrieval_twoEmbedder_usingPad_EOSadded_DEBUG_EOSdouble_END_FOLLOWED.pt")
            raise NotImplementedError
        elif args.use_only_sub_rel_for_retrieval:
            print("INFO: only using sub and rel for retrieval (to get embedding)")
            if args.use_special_tokens_to_split_retrieved_cases:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_atomic_use_only_sub_rel_for_retrieval_specialTokenSplit_twoEmbedder_usingPad_EOSadded_DEBUG_EOSFOLLOWED")
            else:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_atomic_use_only_sub_rel_for_retrieval_twoEmbedder_usingPad_EOSadded_DEBUG_EOSFOLLOWED")
        else:
            if args.use_special_tokens_to_split_retrieved_cases:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_atomic_specialTokenSplit_twoEmbedder_usingPad_EOSadded_DEBUG_EOSFOLLOWED.pt")
            else:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_atomic_twoEmbedder_usingPad_EOSadded_DEBUG_EOSFOLLOWED.pt")
    elif args.dataset_selection == 2:
        if args.use_obj_for_retrieval:
            print("INFO: using obj during retrieval")
            raise NotImplementedError
        elif args.use_only_sub_rel_for_retrieval:
            print("INFO: only using sub and rel for retrieval (to get embedding)")
            if args.if_use_relation_for_shakes:
                if args.use_special_tokens_to_split_retrieved_cases:
                    path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_dataset_shakespear_use_only_sub_rel_for_retrieval_specialTokenSplit_useRelation.pt")
                else:
                    path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_dataset_shakespear_use_only_sub_rel_for_retrieval_useRelation.pt")
            else:
                if args.use_special_tokens_to_split_retrieved_cases:
                    path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_dataset_shakespear_use_only_sub_rel_for_retrieval_specialTokenSplit.pt")
                else:
                    path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_dataset_shakespear_use_only_sub_rel_for_retrieval.pt")
        else:
            if args.if_use_relation_for_shakes:
                if args.use_special_tokens_to_split_retrieved_cases:
                    path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_dataset_shakespear_specialTokenSplit_useRelation.pt")
                else:
                    path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_dataset_shakespear_useRelation.pt")
            else:
                if args.use_special_tokens_to_split_retrieved_cases:
                    path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_dataset_shakespear_specialTokenSplit.pt")
                else:
                    path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_dataset_shakespear.pt")
    elif args.dataset_selection == 3:
        if args.use_obj_for_retrieval:
            raise Exception
        elif args.use_only_sub_rel_for_retrieval:
            print("INFO: only using sub and rel for retrieval (to get embedding)")
            if args.use_special_tokens_to_split_retrieved_cases:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_e2e_use_only_sub_rel_for_retrieval_specialTokenSplit.pt")
            else:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_e2e_use_only_sub_rel_for_retrieval.pt")
        else:
            if args.use_special_tokens_to_split_retrieved_cases:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_e2e_specialTokenSplit.pt")
            else:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_e2e.pt")
    elif args.dataset_selection == 4 or args.dataset_selection == 5 or args.dataset_selection == 6:
        if args.dataset_selection == 4:
            dataset_abrev_name = "sentiment"
        elif args.dataset_selection == 5:
            dataset_abrev_name = "financial"
        elif args.dataset_selection == 6:
            dataset_abrev_name = "yelp"
        else:
            raise NotImplementError
        if args.use_obj_for_retrieval:
            raise Exception
        elif args.use_only_sub_rel_for_retrieval:
            print("INFO: only using sub and rel for retrieval (to get embedding)")
            if args.use_special_tokens_to_split_retrieved_cases:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_" + dataset_abrev_name + "_use_only_sub_rel_for_retrieval_specialTokenSplit.pt")
            else:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_" + dataset_abrev_name + "_use_only_sub_rel_for_retrieval.pt")
        else:
            if args.use_special_tokens_to_split_retrieved_cases:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_" + dataset_abrev_name + "_specialTokenSplit.pt")
            else:
                path_tensor_datasets = os.path.join(args.dataStore_dir, "ECBRF_shared_data_for_reuse/", "tensor_datasets_" + dataset_abrev_name + ".pt")

    # comment this out so that new gpt2 experiment will generate new corresponding tensor_datasets; 9/1/2021
    # if args.generator_model_type != "gpt2-lmhead":
    path_tensor_datasets = path_tensor_datasets.split('.')
    assert(len(path_tensor_datasets) == 2)
    path_tensor_datasets[0] += "_" + args.generator_model_type.split('-')[0]
    path_tensor_datasets = '.'.join(path_tensor_datasets)
    if args.if_not_adding_special_relation_tokens:
        path_tensor_datasets = path_tensor_datasets.split('.')
        assert(len(path_tensor_datasets) == 2)
        path_tensor_datasets[0] += "_noSpecialRelationToken"
        path_tensor_datasets = '.'.join(path_tensor_datasets)
    if args.subset_selection >= 0:
        path_tensor_datasets = path_tensor_datasets.split('.')
        assert(len(path_tensor_datasets) == 2)
        path_tensor_datasets[0] += "_subset_" + str(args.subset_selection) + args.additional_sampling_method_name + args.additional_sample_name
        path_tensor_datasets = '.'.join(path_tensor_datasets)
    else:
        # add args.additional_sampling_method_name
        if args.if_use_nshot_data:
            path_tensor_datasets = path_tensor_datasets.split('.')
            assert(len(path_tensor_datasets) == 2)
            path_tensor_datasets[0] += args.additional_sampling_method_name
            path_tensor_datasets = '.'.join(path_tensor_datasets)
    print("path_tensor_datasets: ", path_tensor_datasets)
    return path_tensor_datasets


# use_only_sub_rel_for_retrieval: for if_double_retrieval
def find_path_sample_ckb_dict(args, use_only_sub_rel_for_retrieval=False):
    if args.dataset_selection == 0:
        if args.use_only_sub_rel_for_retrieval or use_only_sub_rel_for_retrieval:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_use_only_sub_rel_for_retrieval_conceptnet.pt')
        else:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_conceptnet.pt')
    elif args.dataset_selection == 1:
        if args.use_only_sub_rel_for_retrieval or use_only_sub_rel_for_retrieval:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_use_only_sub_rel_for_retrieval_atomic.pt')
        else:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_atomic.pt')
    elif args.dataset_selection == 2:
        if args.use_only_sub_rel_for_retrieval or use_only_sub_rel_for_retrieval:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_use_only_sub_rel_for_retrieval_shakespear.pt')
        else:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_shakespear.pt')
    elif args.dataset_selection == 3:
        if args.use_only_sub_rel_for_retrieval or use_only_sub_rel_for_retrieval:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_use_only_sub_rel_for_retrieval_e2e.pt')
        else:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_e2e.pt')
    elif args.dataset_selection == 4:
        if args.use_only_sub_rel_for_retrieval or use_only_sub_rel_for_retrieval:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_use_only_sub_rel_for_retrieval_sentiment.pt')
        else:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_sentiment.pt')
    elif args.dataset_selection == 5:
        if args.use_only_sub_rel_for_retrieval or use_only_sub_rel_for_retrieval:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_use_only_sub_rel_for_retrieval_financial.pt')
        else:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_financial.pt')
    elif args.dataset_selection == 6:
        if args.use_only_sub_rel_for_retrieval or use_only_sub_rel_for_retrieval:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_use_only_sub_rel_for_retrieval_yelp.pt')
        else:
            path_sample_ckb_dict = os.path.join(args.dataStore_dir, 'ECBRF_shared_data_for_reuse/', 'sample_ckb_dict_dpr_for_bothFro_retriever_yelp.pt')
    if args.retriever_model_type != "dpr":
        path_sample_ckb_dict = path_sample_ckb_dict.split('.')
        assert len(path_sample_ckb_dict) == 2
        path_sample_ckb_dict[0] += args.retriever_model_type
        path_sample_ckb_dict = '.'.join(path_sample_ckb_dict)
    if args.subset_selection >= 0:
        path_sample_ckb_dict = path_sample_ckb_dict.split('.')
        assert len(path_sample_ckb_dict) == 2
        path_sample_ckb_dict[0] += "_subset_" + str(args.subset_selection) + args.additional_sampling_method_name + args.additional_sample_name
        path_sample_ckb_dict = '.'.join(path_sample_ckb_dict)
    else:
        # add args.additional_sampling_method_name
        if args.if_use_nshot_data:
            path_sample_ckb_dict = path_sample_ckb_dict.split('.')
            assert len(path_sample_ckb_dict) == 2
            path_sample_ckb_dict[0] += args.additional_sampling_method_name
            path_sample_ckb_dict = '.'.join(path_sample_ckb_dict)
    return path_sample_ckb_dict
