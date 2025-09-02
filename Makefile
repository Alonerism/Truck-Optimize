api:
	poetry run uvicorn app.api:app --reload --port 8000

ui-v1:
	cd ui-v1 && npm i && npm run dev

ui-lovable:
	cd ui-lovable && npm i && npm run dev

.PHONY: api ui-v1 ui-lovable
