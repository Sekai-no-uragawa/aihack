/* DOCSTRING
| Этот запрос создает базу данных необходимую для выгрузки данных из БД
| и передачи их в модель для обучения выдачи кода ВЭД по описанию
| товара переданного в таможню для декларации.
*/

CREATE TABLE Submitted_declarations (
	id_declarations INT(16),
	name_applicant VARCHAR(250),
	date_of_filing DATE,
	description VARCHAR(2000),
	id_code VARCHAR(2000),
	custom_tax FLOAT(20, 2),
	type_of_cargo VARCHAR(100),
	size VARCHAR(50),
	amount INT(20),
	price_per_position FLOAT(20, 2),
	form_certificate VARCHAR(50),
	num_certificate VARCHAR(100),
	date_certificate DATE,
	form_quality_certificate VARCHAR(50),
	num_quality_certificate VARCHAR(100),
	date_quality_certificate DATE,
	type_other_documents VARCHAR(50),
	num_other_documents VARCHAR(100),
	date_other_documents DATE
);

CREATE TABLE Guide_VED (
	id_code INT(10),
	russian_code VARCHAR(10),
	internarional_code VARCHAR(6),
	last_code VARCHAR(4),
	name VARCHAR(2000),
	tax_percent FLOAT(2, 3),
	min_price_for_one_position FLOAT(2, 2)
);

CREATE TABLE Accept_declarations (
	id_declarations INT(10),
	id_declaration INT(10),
	date_accepted DATE,
	accept_code VARCHAR(10),
	accept_tax FLOAT(25, 2)
);

CREATE TABLE Employee (
	id_employee INT(10),
	first_name VARCHAR(50),
	last_name VARCHAR(50),
	employment_date DATE
);

CREATE TABLE Cancel_declarations (
	id_declaration INT(10),
	id_employee INT(10),
	date DATE,
	correct_code INT(1),
	correct_tax INT(1)
);