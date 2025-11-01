-- Example PostgreSQL schema for Autonomous Systems outputs
CREATE TABLE quasim_kernel_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  vertical TEXT NOT NULL,
  seed INTEGER NOT NULL,
  payload JSONB NOT NULL,
  result JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
