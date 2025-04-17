# snowflake-connector-python

Personal Snowflake Connector Examples

## Installing the project

1. Clone this repository
2. In your directory create a virtual enviroment for the packages of this project
    ```bash
    python3 -m venv .venv
    ```

3. Activate your virtual environment
    ```bash
    source .venv/bin/activate
    ```
4. Install the necessary packages using requirements file

    ```bash
    pip3 install -r requirements.txt
    ```

## Setting Up

To test the script, you will need a machine user with an rsa_key in order to connecto to Snowflake, the content of the yaml file has to be:


    ```YML
    snowflake_dbt:
        account: SNOWFLAKE_ACCOUNT
        host: SNOWFALKE_HOSTNAME
        user: SNOWFLAKE_USERNAME
        database: SNOWFLAKE_DATABASE_NAME
        schema: SNOWFLAKE_SCHEMA_NAME
        warehouse: SNOWFLAKE_WAREHOUSE_NAME
        role: SNOWFLAKE_ROLE_NAME
        rsa_key: |
            -----BEGIN PRIVATE KEY-----
            ...
            -----END PRIVATE KEY-----
    ```