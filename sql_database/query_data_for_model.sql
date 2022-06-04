/* DOCSTRING
| Этот запрос вернет данные которые нужны для обучения модели.
| Все непонятные места будут прокомментированы.
*/

SELECT
	t1.id_code AS VED,
	t1.description AS OPISANIE,
	t2.date_accepted AS DATE,
	t3.name AS OPISANIE_VED
FROM Submitted_declarations AS t1
INNER JOIN Accept_declarations AS t2
ON t1.id_declaration=t2.id_declaration
INNER JOIN Guide_VED AS t3
ON t1.id_code=t3.id_code
WHERE t3.name IS NOT NULL -- по условию если дата не пустая, то мы можем быть уверены что данные валидированы
	AND t3.name>TO_DATE(&variable, 'dd.mm.yyyy') -- если понадобится обучить данные заново после изменения закона и ВЭДов