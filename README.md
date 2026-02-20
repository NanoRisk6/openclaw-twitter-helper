# Open Claw Twitter Helper

Friendly CLI for posting tweets and threads with interactive setup and a true browser OAuth wizard.

## Quickstart (Recommended)

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py setup
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py app-settings
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py post --text "hello from Open Claw"
```

`setup` now prompts to immediately launch OAuth2 browser login so users can generate OAuth2 access/refresh tokens in the same flow.
`setup` also auto-sets OAuth scopes to `tweet.read tweet.write users.read offline.access` so users do not need to choose scopes.
`auth-login` now runs `doctor` automatically after token exchange.

## One-Command Mode For Open Claw

When you want Open Claw (or a subagent) to fully handle check + repair + post:

```bash
/Users/matthew/openclaw-twitter-helper/run-twitter-helper --text "Open Claw status update"
```

Equivalent direct command:

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py run-twitter-helper --text "Open Claw status update"
```

This command will:
- locate the helper + env paths,
- run readiness checks,
- launch OAuth browser repair if auth is broken,
- post a unique public tweet by appending a UTC timestamp suffix.

## Twitter App Settings (What to choose)

Run this to print exact values from your current env:

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py app-settings
```

Use these settings in Twitter Developer Portal:

- `Type of App`: `Web App, Automated App or Bot` (Confidential client)
- `App permissions`: `Read and write`
- `OAuth 2.0 Client ID` -> `TWITTER_CLIENT_ID`
- `OAuth 2.0 Client Secret` -> `TWITTER_CLIENT_SECRET`
- `Callback URI / Redirect URL` -> `TWITTER_REDIRECT_URI`
- `Website URL` -> `TWITTER_WEBSITE_URL`

## Flexible Redirect Input

In `setup`, you can enter redirect URI in flexible forms and it will normalize for you:

- `127.0.0.1:3000`
- `localhost:3000/callback`
- `http://127.0.0.1:8080`
- Full URL with path

The helper converts these into a valid redirect URI (adds scheme/path as needed).

## Commands

Setup app config:

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py setup
```

Show exact app settings:

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py app-settings
```

Run browser OAuth wizard (gets/saves access + refresh tokens):

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py auth-login
```
Use `--skip-doctor` if you do not want automatic diagnostics after login.
Use `--auto-post` to post a unique confirmation tweet automatically after successful login+doctor.

Note: OAuth1 keys/tokens from the portal are not used for this flow. This helper uses OAuth2 tokens generated via the consent link.

Run diagnostics:

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py doctor
```

Machine-readable status for Open Claw:

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py openclaw-status
```

Human walkthrough:

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py walkthrough
```

Post one tweet:

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py post --text "hello from Open Claw"
```

Post a thread (`---` between tweets):

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py thread --file /Users/matthew/openclaw-twitter-helper/examples/thread.txt
```

Open Claw integration post:

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py openclaw-autopost --text "Open Claw status update"
```

Preview final output without posting:

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py openclaw-autopost --text "Open Claw status update" --dry-run
```

Open Claw readiness check (doctor-style):

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py openclaw
```

Machine-readable readiness:

```bash
python3 /Users/matthew/openclaw-twitter-helper/src/twitter_helper.py openclaw --json
```
