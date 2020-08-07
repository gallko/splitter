#pragma once

#include <vector>
#include <memory>

#ifndef IN
#   define IN
#endif
#ifndef OUT
#   define OUT
#endif

// ISplitter интерфейс
class ISplitter {
    virtual bool SplitterInfoGet(OUT int *_pnMaxBuffers, OUT int *_pnMaxClients) = 0;

// Кладём данные в очередь. Если какой-то клиент не успел ещё забрать свои данные, и количество буферов (задержка) для него больше максимального значения, то ждём пока не освободятся буфера (клиент заберет данные) в течении _nTimeOutMsec. Если по истечению времени данные так и не забраны, то удаляем старые данные для этого клиента, добавляем новые (по принципу FIFO) (*). Возвращаем код ошибки, который дает понять что один или несколько клиентов “пропустили” свои данные.
    virtual int SplitterPut(IN const std::shared_ptr <std::vector<uint8_t>> &_pVecPut, IN int _nTimeOutMsec) = 0;

// Сбрасываем все буфера, прерываем все ожидания.
    virtual int SplitterFlush() = 0;

// Добавляем нового клиента - возвращаем уникальный идентификатор клиента.
    virtual bool SplitterClientAdd(OUT int *_pnClientID) = 0;

// Удаляем клиента по идентификатору, если клиент находиться в процессе ожидания буфера, то прерываем ожидание.
    virtual bool SplitterClientRemove(IN int _nClientID) = 0;

// Перечисление клиентов, для каждого клиента возвращаем его идентификатор и количество буферов в очереди (задержку) для этого клиента.
    virtual bool SplitterClientGetCount(OUT int *_pnCount) = 0;

    virtual bool SplitterClientGetByIndex(IN int _nIndex, OUT int *_pnClientID, OUT int *_pnLatency) = 0;

// По идентификатору клиента запрашиваем данные, если данных пока нет, то ожидаем _nTimeOutMsec пока не будут добавлены новые данные, в случае превышения времени ожидания - возвращаем ошибку.
    virtual int SplitterGet(IN int _nClientID, OUT std::shared_ptr <std::vector<uint8_t>> &_pVecGet, IN int _nTimeOutMsec) = 0;

// Закрытие объекта сплиттера - все ожидания должны быть прерваны все вызовы возвращают соответствующую ошибку.
    virtual void SplitterClose() = 0;

//(*)Пусть количество буферов (максимальная задержка) равно 2. Мы положили в сплиттер буфера 1,2,3,4,5,6,7,8,9,10 (с интервалом в 100 msec, максимальное время ожидания в SplitterPut - 50 msec).
//- Клиент 1 сразу получил 1,2,3 а затем 500 msec “спал”, то после того как проснется он должен получить 7,8,9,10 (4, 5, 6 будут потеряны)
//- Остальные клиенты должны в это время получить все буфера 1,2,3,4,5,6,7,8,9,10 с максимальной задержкой 50 msec (для буферов 6, 7, 8,).
};

// Создание объекта сплиттера - задаётся максимальное количество буферов в очереди, и максимальное количество клиентов.
std::shared_ptr<ISplitter> SplitterCreate(IN int _nMaxBuffers, IN int _nMaxClients);
