#
#
#

.PHONY: __dummy__ test
__dummy__:
	@echo "make {test}"

test:
	python3 -m unittest discover tests