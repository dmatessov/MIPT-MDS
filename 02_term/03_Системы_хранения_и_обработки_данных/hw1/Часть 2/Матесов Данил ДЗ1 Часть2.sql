CREATE SCHEMA `healthcare-analytics`;

CREATE  TABLE employer_category ( 
	id                   INT  NOT NULL   AUTO_INCREMENT  PRIMARY KEY,
	employer_category    VARCHAR(100)  NOT NULL     
 ) engine=InnoDB;

CREATE  TABLE health_camp_detail ( 
	health_camp_id       INT  NOT NULL     PRIMARY KEY,
	camp_start_date      DATE  NOT NULL     ,
	camp_end_date        DATE  NOT NULL     ,
	category1            INT  NOT NULL     ,
	category2            CHAR(1)  NOT NULL     ,
	category3            INT  NOT NULL     
 ) engine=InnoDB;

CREATE  TABLE patient_profile ( 
	patient_id           INT  NOT NULL     PRIMARY KEY,
	online_follower      BOOLEAN  NOT NULL DEFAULT (0)    ,
	linkedin_shared      BOOLEAN  NOT NULL DEFAULT (0)    ,
	twitter_shared       BOOLEAN  NOT NULL DEFAULT (0)    ,
	facebook_shared      BOOLEAN  NOT NULL DEFAULT (0)    ,
	income               INT       ,
	education_score      DECIMAL       ,
	age                  INT       ,
	first_interaction    DATE  NOT NULL     ,
	city_type            CHAR(1)       ,
	employer_category    INT       
 ) engine=InnoDB;

CREATE  TABLE second_health_camp_attended ( 
	patient_id           INT  NOT NULL     ,
	health_camp_id       INT  NOT NULL     ,
	`health score`       DECIMAL  NOT NULL     
 ) engine=InnoDB;

CREATE  TABLE third_health_camp_attended ( 
	patient_id           INT  NOT NULL     ,
	health_camp_id       INT  NOT NULL     ,
	number_of_stall_visited INT  NOT NULL     ,
	last_stall_visited_number INT  NOT NULL     
 ) engine=InnoDB;

CREATE  TABLE attendance ( 
	patient_id           INT  NOT NULL     ,
	health_camp_id       INT  NOT NULL     ,
	registration_date    DATE  NOT NULL     ,
	var1                 INT  NOT NULL     ,
	var2                 INT  NOT NULL     ,
	var3                 INT  NOT NULL     ,
	var5                 INT  NOT NULL     
 ) engine=InnoDB;

CREATE  TABLE first_health_camp_attended ( 
	patient_id           INT  NOT NULL     ,
	health_camp_id       INT  NOT NULL     ,
	donation             INT  NOT NULL     ,
	health_score         DECIMAL  NOT NULL     
 ) engine=InnoDB;

ALTER TABLE attendance ADD CONSTRAINT fk_attendance_patient_profile FOREIGN KEY ( patient_id ) REFERENCES patient_profile( patient_id ) ON DELETE NO ACTION ON UPDATE NO ACTION;

ALTER TABLE attendance ADD CONSTRAINT fk_attendance_health_camp_detail FOREIGN KEY ( health_camp_id ) REFERENCES health_camp_detail( health_camp_id ) ON DELETE NO ACTION ON UPDATE NO ACTION;

ALTER TABLE first_health_camp_attended ADD CONSTRAINT fk_first_health_camp_attended_patient_profile FOREIGN KEY ( patient_id ) REFERENCES patient_profile( patient_id ) ON DELETE NO ACTION ON UPDATE NO ACTION;

ALTER TABLE first_health_camp_attended ADD CONSTRAINT fk_first_health_camp_attended_health_camp_detail FOREIGN KEY ( health_camp_id ) REFERENCES health_camp_detail( health_camp_id ) ON DELETE NO ACTION ON UPDATE NO ACTION;

ALTER TABLE patient_profile ADD CONSTRAINT fk_patient_profile_employer_category FOREIGN KEY ( employer_category ) REFERENCES employer_category( id ) ON DELETE NO ACTION ON UPDATE NO ACTION;

ALTER TABLE second_health_camp_attended ADD CONSTRAINT fk_second_health_camp_attended_patient_profile FOREIGN KEY ( patient_id ) REFERENCES patient_profile( patient_id ) ON DELETE NO ACTION ON UPDATE NO ACTION;

ALTER TABLE second_health_camp_attended ADD CONSTRAINT fk_second_health_camp_attended_health_camp_detail FOREIGN KEY ( health_camp_id ) REFERENCES health_camp_detail( health_camp_id ) ON DELETE NO ACTION ON UPDATE NO ACTION;

ALTER TABLE third_health_camp_attended ADD CONSTRAINT fk_third_health_camp_attended_patient_profile FOREIGN KEY ( patient_id ) REFERENCES patient_profile( patient_id ) ON DELETE NO ACTION ON UPDATE NO ACTION;

ALTER TABLE third_health_camp_attended ADD CONSTRAINT fk_third_health_camp_attended_health_camp_detail FOREIGN KEY ( health_camp_id ) REFERENCES health_camp_detail( health_camp_id ) ON DELETE NO ACTION ON UPDATE NO ACTION;

