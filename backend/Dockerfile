version: '3.8'

services:
  # Backend API (FastAPI)
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: deepguard-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro  # Mount models as read-only
      - ./backend:/app           # Hot reload in development
    environment:
      - PYTHONUNBUFFERED=1
    networks:
      - deepguard-network
    restart: unless-stopped

  # Frontend (React)
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: deepguard-frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - backend
    networks:
      - deepguard-network
    restart: unless-stopped

networks:
  deepguard-network:
    driver: bridge

volumes:
  models: