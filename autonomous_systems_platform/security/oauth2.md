# OAuth2 JWT Authentication

The backend validates JWT tokens in the Authorization header.

## Configuration

- Set `JWT_SECRET` environment variable
- Tokens must include `sub` (user ID) and `exp` (expiration)

## Example

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/kernel
```
