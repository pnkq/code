#!/bin/bash

mvn clean
mvn compile war:war
mvn cargo:redeploy
