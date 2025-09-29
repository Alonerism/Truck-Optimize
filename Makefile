api:
	poetry run uvicorn app.api:app --reload --port 8000

ui-lovable:
	cd ui-lovable && npm i && npm run dev

.PHONY: api ui-lovable
