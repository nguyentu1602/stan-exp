submit:
	python submit.py

clean:
	mv assignment*.json submitresults

envexport:
	conda env export | grep -v "^prefix: " > environment.yml

envcreate:
	conda env create -f environment.yml

testhq:
	nosetests -v search_submission_tests.py:TestPriorityQueue

testbfs:
	nosetests --nocapture -v search_unit_tests.py:SearchUnitTests.test_bfs_romania

