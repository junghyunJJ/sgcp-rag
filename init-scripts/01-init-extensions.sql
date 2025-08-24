-- PostgreSQL will automatically create the database specified in POSTGRES_DB
-- This script adds any additional initialization

-- Enable pgvector extension if needed
CREATE EXTENSION IF NOT EXISTS vector;

-- You can add more initialization SQL here
-- For example, creating tables, indexes, etc.