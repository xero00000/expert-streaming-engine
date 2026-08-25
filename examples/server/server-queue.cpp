#include "server-task.h"
#include "server-queue.h"
#include "server-common.h"

#include "log.h"
#include <chrono>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)


int server_queue::post(server_task task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!accepting) {
        throw std::runtime_error("server task queue is shutting down");
    }
    if (task.id == -1) {
        task.id = id++;
        //LOG_VERBOSE("new task id", { {"new_id", task.id} });
        QUE_DBG("new task, id = %d\n", task.id);
    }
    queue_tasks.push_back(std::move(task));
    condition_tasks.notify_one();
    return task.id;
}

std::vector<server_task> server_queue::take_pending_tasks(int id_target) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    size_t count = 0;
    for (const auto & task : queue_tasks) count += task.id == id_target;
    for (const auto & task : queue_tasks_deferred) count += task.id == id_target;

    std::vector<server_task> removed;
    removed.reserve(count);
    const auto extract = [&](auto & queue) {
        for (auto it = queue.begin(); it != queue.end();) {
            if (it->id == id_target) {
                removed.push_back(std::move(*it));
                it = queue.erase(it);
            } else {
                ++it;
            }
        }
    };
    extract(queue_tasks);
    extract(queue_tasks_deferred);
    return removed;
}

std::vector<server_task> server_queue::take_all_pending_tasks() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    std::vector<server_task> removed;
    removed.reserve(queue_tasks.size() + queue_tasks_deferred.size());
    for (auto & task : queue_tasks) removed.push_back(std::move(task));
    for (auto & task : queue_tasks_deferred) removed.push_back(std::move(task));
    queue_tasks.clear();
    queue_tasks_deferred.clear();
    return removed;
}

// multi-task version of post()
int server_queue::post(std::vector<server_task>&& tasks, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!accepting) {
        throw std::runtime_error("server task queue is shutting down");
    }
    std::vector<int> task_ids;
    task_ids.reserve(tasks.size());
    for (auto & task : tasks) {
        if (task.id == -1) {
            task.id = id++;
        }
        task_ids.push_back(task.id);
    }

    size_t inserted = 0;
    try {
        for (auto& task : tasks) {
        QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int)tasks.size(), front);
        if (front) {
            queue_tasks.push_front(std::move(task));
        }
        else {
            queue_tasks.push_back(std::move(task));
        }
            inserted++;
        }
    } catch (...) {
        // Batch posting is failure-atomic. HTTP-side transient lease guards can
        // safely release every task lease when this call throws because no
        // partially enqueued task survives.
        queue_tasks.erase(
            std::remove_if(
                queue_tasks.begin(), queue_tasks.end(),
                [&](const server_task & queued) {
                    return std::find(
                        task_ids.begin(), task_ids.begin() + inserted,
                        queued.id) != task_ids.begin() + inserted;
                }),
            queue_tasks.end());
        throw;
    }
    condition_tasks.notify_one();
    return 0;
}

void server_queue::defer(server_task&& task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    queue_tasks_deferred.push_back(std::move(task));
}

int server_queue::get_new_id() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!accepting) {
        throw std::runtime_error("server task queue is shutting down");
    }
    int new_id = id++;
    //LOG_VERBOSE("new task id", { {"new_id", new_id} });
    QUE_DBG("new task, id = %d\n", id);
    return new_id;
}

void server_queue::notify_slot_changed() {
    // move deferred tasks back to main loop
    std::unique_lock<std::mutex> lock(mutex_tasks);
    for (auto& task : queue_tasks_deferred) {
        queue_tasks.push_back(std::move(task));
    }
    queue_tasks_deferred.clear();
}

void server_queue::on_new_task(std::function<void(server_task&&)> callback) {
    callback_new_task = std::move(callback);
}


void server_queue::start_loop() {
    while (true) {
        LOG_VERBOSE("new task may arrive", {});

        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (queue_tasks.empty()) {
                lock.unlock();
                break;
            }
            server_task task = std::move(queue_tasks.front());
            queue_tasks.pop_front();
            lock.unlock();
            //LOG_VERBOSE("callback_new_task", { {"id_task", task.id} });
            callback_new_task(std::move(task));
        }

        LOG_VERBOSE("update_multitasks", {});

        // check if we have any finished multitasks
        auto queue_iterator = queue_multitasks.begin();
        while (queue_iterator != queue_multitasks.end()) {
            if (queue_iterator->subtasks_remaining.empty()) {
                // all subtasks done == multitask is done
                server_task_multi current_multitask = *queue_iterator;
                callback_finish_multitask(current_multitask);
                // remove this multitask
                queue_iterator = queue_multitasks.erase(queue_iterator);
            }
            else {
                ++queue_iterator;
            }
        }

        // all tasks in the current loop is processed, slots data is now ready
        LOG_VERBOSE("callback_update_slots", {});

        callback_update_slots();

        LOG_VERBOSE("wait for new task", {});
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (queue_tasks.empty()) {
                if (!running) {
                    LOG_VERBOSE("ending start_loop", {});
                    return;
                }
                condition_tasks.wait(lock, [&] {
                    return (!queue_tasks.empty() || !running);
                    });
            }
        }
    }
}


void server_queue::add_multitask(int id_multi, std::vector<int>& sub_ids) {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    server_task_multi multi;
    multi.id = id_multi;
    std::copy(sub_ids.begin(), sub_ids.end(), std::inserter(multi.subtasks_remaining, multi.subtasks_remaining.end()));
    queue_multitasks.push_back(multi);
}


void server_queue::update_multitask(int id_multi, int id_sub, server_task_result& result) {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    for (auto& multitask : queue_multitasks) {
        if (multitask.id == id_multi) {
            multitask.subtasks_remaining.erase(id_sub);
            multitask.results.push_back(result);
        }
    }
}


void server_response::add_waiting_task_id(int id_task) {
    SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int)waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.insert(id_task);
}

void server_response::add_waiting_tasks(const std::vector<server_task>& tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);
    auto candidate = waiting_task_ids;
    for (const auto& task : tasks) {
        SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", task.id, (int)waiting_task_ids.size());
        candidate.insert(task.id);
    }
    waiting_task_ids.swap(candidate);
}

void server_response::remove_waiting_task_id(int id_task) {
    //LOG_VERBOSE("remove waiting for task id", { {"id_task", id_task} });
    QUE_DBG("remove waiting for task id, id = %d\n", id_task);
    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.erase(id_task);
}


server_task_result server_response::recv(int id_task) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&] {
            return !queue_results_legacy.empty() || !running;
            });

        for (int i = 0; i < (int)queue_results_legacy.size(); i++) {
            if (queue_results_legacy[i].id == id_task) {
                assert(queue_results_legacy[i].id_multi == -1);
                server_task_result res = queue_results_legacy[i];
                queue_results_legacy.erase(queue_results_legacy.begin() + i);
                return res;
            }
        }
        if (!running) {
            throw std::runtime_error("server response queue is shutting down");
        }
    }

    // should never reach here
}

// same as recv(), but have timeout in seconds
// if timeout is reached, nullptr is returned
server_task_result_ptr server_response::recv_with_timeout(const std::unordered_set<int>& id_tasks, int timeout) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout);
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (int i = 0; i < (int)queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }

        std::cv_status cr_res = condition_results.wait_until(lock, deadline);
        if (!running) {
            SRV_DBG("%s : queue result stop\n", __func__);
            return nullptr;
        }
        if (cr_res == std::cv_status::timeout) {
            return nullptr;
        }
    }

    // should never reach here
}

bool server_response::is_running() {
    std::lock_guard<std::mutex> lock(mutex_results);
    return running;
}

void server_response::remove_waiting_task_ids(const std::unordered_set<int>& id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto& id_task : id_tasks) {
        SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int)waiting_task_ids.size());
        waiting_task_ids.erase(id_task);
    }
}

void server_response::send(server_task_result result) {
    //LOG_VERBOSE("send new result", { {"id_task", result.id} });
    QUE_DBG("send new result, id = %d\n", result.id);
    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto& id_task : waiting_task_ids) {
        // LOG_TEE("waiting task id %i \n", id_task);
        // for now, tasks that have associated parent multitasks just get erased once multitask picks up the result
        if (result.id_multi == id_task) {
            //LOG_VERBOSE("callback_update_multitask", { {"id_task", id_task} });
            QUE_DBG("callback_update_multitask, id = %d\n", id_task);
            callback_update_multitask(id_task, result.id, result);
            continue;
        }

        if (result.id == id_task) {
            //LOG_VERBOSE("queue_results_legacy.push_back", { {"id_task", id_task} });
            QUE_DBG("queue_results.push_back, id = %d\n", id_task);
            queue_results_legacy.push_back(std::move(result));
            condition_results.notify_all();
            return;
        }
    }
}

// Send a new result to a waiting id_task
void server_response::send(server_task_result_ptr&& result) {
    SRV_DBG("sending result for task id = %d\n", result->id);

    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto& id_task : waiting_task_ids) {
        if (result->id == id_task) {
            SRV_DBG("task id = %d pushed to result queue\n", result->id);

            queue_results.emplace_back(std::move(result));
            condition_results.notify_all();
            return;
        }
    }
}
