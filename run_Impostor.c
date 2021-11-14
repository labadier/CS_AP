#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h> 

struct info{
    int index;
    char lang[3];
    char model[15];
    char task[15];
};


void * handler( void *arg){

    struct info model = *((struct info *)arg);

    char command[256];
    char coef[8];
    strcpy(coef, "0.");
    for(int i = 10; i <= 90; i+= 5){

        coef[2] = i/10 + '0';
        coef[3] = i%10 + '0';
        
        strcpy(command, "python main.py -l ");
        strcat(command, model.lang);
        strcat(command, " -mode Impostor -dp data/profiling/");
        strcat(command, model.task);
        strcat(command, " -rp ");
        strcat(command, coef);
        strcat(command, " -metric cosine -rep ");
        strcat(command, model.model );
        strcat(command, " -task ");
        strcat(command, model.task);
        strcat(command, " >> experiments/");
        strcat(command, model.model);
        strcat(command, "_");
        strcat(command, model.task);
        strcat(command, "_");
        strcat(command, model.lang);
        printf("%d %s\n", model.index, command);
        system(command);
    }

	pthread_exit(NULL);
}

char models[5][15];
char tasks[5][15];
char lang[5][15];

void main(){

    pthread_t tid[12];
    strcpy(models[0], "lstm");
    strcpy(models[1], "gcn");

    strcpy(tasks[0], "faker");
    strcpy(tasks[1], "hater");
    strcpy(tasks[2], "bot");

    strcpy(lang[0], "EN");
    strcpy(lang[1], "ES");

    struct info send[12];
    
    int index = 0;
    for(int i = 0; i < 3; i++){

    
      for(int j = 0; j < 2; j++){
          
          for(int k = 0; k < 2; k++){
            
            send[index].index = index;
            strcpy(send[index].model, models[j]);
            strcpy(send[index].lang, lang[k]);
            strcpy(send[index].task, tasks[i]);

            if(pthread_create(&tid[index], NULL, handler, &send[index]) != 0)
              printf("Error\n");
            index++;
          }
      }
    }
    
    for (int i = 0; i < 12; i++)
       pthread_join(tid[i], NULL);

}