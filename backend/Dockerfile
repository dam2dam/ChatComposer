FROM azul/zulu-openjdk:11

ARG JAR_FILE=build/libs/ChatComposer-0.0.1-SNAPSHOT.jar
COPY ${JAR_FILE} app.jar

EXPOSE 8080
ENV TERM=xterm

CMD ["java", "-jar", "app.jar"]